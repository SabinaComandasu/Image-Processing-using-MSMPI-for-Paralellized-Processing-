#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <chrono>
#include <sys/stat.h>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

std::vector<unsigned char> resize_rows(const std::vector<unsigned char>& data, int width, int inHeight, int outHeight, int channels) {
    std::vector<unsigned char> out(width * outHeight * channels);
    for (int y = 0; y < outHeight; ++y) {
        int srcY = y * inHeight / outHeight;
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                out[(y * width + x) * channels + c] =
                    data[(srcY * width + x) * channels + c];
            }
        }
    }
    return out;
}

void apply_filter(unsigned char* data, int width, int height, int channels, const std::string& filterType) {
    if (filterType == "invert") {
        for (int i = 0; i < width * height * channels; ++i)
            data[i] = 255 - data[i];
    }
    else if (filterType == "brightness") {
        for (int i = 0; i < width * height * channels; ++i) {
            int val = data[i] + 50;
            data[i] = (val > 255) ? 255 : val;
        }
    }
    else if (filterType == "contrast") {
        float factor = 1.2f;
        for (int i = 0; i < width * height * channels; ++i) {
            int val = static_cast<int>((data[i] - 128) * factor + 128);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            data[i] = static_cast<unsigned char>(val);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto startTime = std::chrono::high_resolution_clock::now();

    std::string inputPath, outputPath, filterType;
    int width = 0, height = 0, channels = 0;
    int newWidth = 0, newHeight = 0;

    std::vector<unsigned char> fullImage;

    if (rank == 0) {
        std::string inFile, outFile;
        std::cout << "Enter input image filename (e.g., avatar.jpg): ";
        std::getline(std::cin, inFile);
        inputPath = "inputs/" + inFile;
        if (!file_exists(inputPath)) {
            std::cerr << "File not found." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::cout << "Enter output image filename (e.g., output.jpg): ";
        std::getline(std::cin, outFile);
        outputPath = "outputs/" + outFile;

        std::cout << "Enter filter (invert, brightness, contrast): ";
        std::getline(std::cin, filterType);

        std::cout << "Enter new width (0 to keep original): ";
        std::cin >> newWidth;
        std::cout << "Enter new height (0 to keep original): ";
        std::cin >> newHeight;

        unsigned char* raw = stbi_load(inputPath.c_str(), &width, &height, &channels, 0);
        if (!raw) {
            std::cerr << "Failed to load image.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fullImage.assign(raw, raw + width * height * channels);
        stbi_image_free(raw);

        std::cout << "Original size: " << width << "x" << height << " (" << (width * height * channels / 1024.0 / 1024.0) << " MB)\n";

        if (newWidth == 0) newWidth = width;
        if (newHeight == 0) newHeight = height;
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&newWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&newHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rowsPerProc = height / size;
    int remainder = height % size;
    int localInHeight = rowsPerProc + (rank < remainder ? 1 : 0);
    int startRow = rank * rowsPerProc + std::min(rank, remainder);

    int localInBytes = localInHeight * width * channels;
    std::vector<unsigned char> localIn(localInBytes);

    std::vector<int> sendCounts(size), displs(size);
    if (rank == 0) {
        for (int i = 0, offset = 0; i < size; ++i) {
            int rpp = height / size + (i < remainder ? 1 : 0);
            sendCounts[i] = rpp * width * channels;
            displs[i] = offset;
            offset += sendCounts[i];
        }
    }

    MPI_Scatterv(rank == 0 ? fullImage.data() : nullptr, sendCounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
        localIn.data(), localInBytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int localOutHeight = newHeight * localInHeight / height;
    std::vector<unsigned char> localResized = resize_rows(localIn, width, localInHeight, localOutHeight, channels);

    apply_filter(localResized.data(), width, localOutHeight, channels, filterType);

    std::vector<int> recvCounts(size), recvDispls(size);
    if (rank == 0) {
        for (int i = 0, offset = 0; i < size; ++i) {
            int inRows = height / size + (i < remainder ? 1 : 0);
            int outRows = newHeight * inRows / height;
            recvCounts[i] = outRows * width * channels;
            recvDispls[i] = offset;
            offset += recvCounts[i];
        }
        fullImage.resize(newWidth * newHeight * channels);
    }

    MPI_Gatherv(localResized.data(), localOutHeight * width * channels, MPI_UNSIGNED_CHAR,
        rank == 0 ? fullImage.data() : nullptr, recvCounts.data(), recvDispls.data(), MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        stbi_write_jpg(outputPath.c_str(), newWidth, newHeight, channels, fullImage.data(), 100);
        std::cout << "Saved resized image to " << outputPath << std::endl;

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "Total processing time: " << duration.count() << " ms" << std::endl;
        std::cout << "Total MPI processes used: " << size << std::endl;
    }

    MPI_Finalize();
    return 0;
}
