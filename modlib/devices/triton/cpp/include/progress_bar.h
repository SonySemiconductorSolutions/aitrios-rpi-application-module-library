/*
 * Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#endif

class ProgressBar {
public:
    explicit ProgressBar(int total, int width = 40)
        : total_(total), width_(width) {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8); // Set UTF-8 code page
#endif
    }

    static std::string bytesToMB(int bytes) {
        double mb = bytes / (1024.0 * 1024.0);
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << mb << " MB";
        return ss.str();
    }

    void update(int current) {
        if (current < 0) current = 0;
        if (current > total_) current = total_;

        double ratio = (total_ > 0) ? double(current) / total_ : 0.0;
        int filled = static_cast<int>(std::floor(ratio * width_));
        double frac = ratio * width_ - filled;

        // Determine partial block index (0=no block, 1=1/8, ..., 7=7/8).
        int partialIndex = static_cast<int>(std::floor(frac * 8 + 0.5));
        if (partialIndex >= 8) {
            partialIndex = 0;
            if (filled < width_) ++filled;
        }

        // Block characters for fractional fills (from 1/8 to full).
        static const char* blocks[] = { " ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█" };
        // Note: index 0 is space (no fill), 8 is full block.

        // Compute empty (rest) width
        int emptyWidth = width_ - filled - (partialIndex > 0 ? 1 : 0);

        // ANSI color codes: 95 = bright magenta/pink, 0 = reset.
        const char* COLOR = "\033[95m";
        const char* RESET = "\033[0m";

        // Print percentage (right-aligned to 3 digits)
        std::cout << "\r";       
        std::cout << std::setw(3) << (int)std::round(ratio * 100.0) << "% ";
        
        // Print progress bar
        std::cout << "[";
        std::cout << COLOR;
        for(int i = 0; i < filled; ++i) {
            std::cout << blocks[8];
        }
        if (partialIndex > 0) {
            std::cout << blocks[partialIndex];
        }
        std::cout << RESET;
        for(int i = 0; i < emptyWidth; ++i) {
            std::cout << " ";
        }
        std::cout << "] " << bytesToMB(current) << "/" << bytesToMB(total_) << std::flush;
    }

private:
    int total_;
    int width_;
};