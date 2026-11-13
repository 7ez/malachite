// SPDX-License-Identifier: BSD-3-Clause
//
// VP9's default probabilities and probability adaptation, as libvpx has them in vp9/common/vp9_entropy.c,
// vp9_entropymode.c, vp9_entropymv.c, vp9/decoder/vp9_dsubexp.c and vpx_dsp/prob.h:
//
// Copyright (c) 2010, The WebM Project authors. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in
//     the documentation and/or other materials provided with the
//     distribution.
//
//   * Neither the name of Google, nor the WebM Project, nor the names
//     of its contributors may be used to endorse or promote products
//     derived from this software without specific prior written
//     permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "vp9prob.h"
#include "bitreader.h"

// init_mode_probs(), vp9_default_coef_probs() and vp9_init_mv_probs()
static const struct vp9_probs defaults = {
    .y_mode = {
        { 65, 32, 18, 144, 162, 194, 41, 51, 98 },
        { 132, 68, 18, 165, 217, 196, 45, 40, 78 },
        { 173, 80, 19, 176, 240, 193, 64, 35, 46 },
        { 221, 135, 38, 194, 248, 121, 96, 85, 29 },
    },
    .uv_mode = {
        { 120, 7, 76, 176, 208, 126, 28, 54, 103 },
        { 48, 12, 154, 155, 139, 90, 34, 117, 119 },
        { 67, 6, 25, 204, 243, 158, 13, 21, 96 },
        { 97, 5, 44, 131, 176, 139, 48, 68, 97 },
        { 83, 5, 42, 156, 111, 152, 26, 49, 152 },
        { 80, 5, 58, 178, 74, 83, 33, 62, 145 },
        { 86, 5, 32, 154, 192, 168, 14, 22, 163 },
        { 85, 5, 32, 156, 216, 148, 19, 29, 73 },
        { 77, 7, 64, 116, 132, 122, 37, 126, 120 },
        { 101, 21, 107, 181, 192, 103, 19, 67, 125 },
    },
    .partition = {
        { 199, 122, 141 },
        { 147, 63, 159 },
        { 148, 133, 118 },
        { 121, 104, 114 },
        { 174, 73, 87 },
        { 92, 41, 83 },
        { 82, 99, 50 },
        { 53, 39, 39 },
        { 177, 58, 59 },
        { 68, 26, 63 },
        { 52, 79, 25 },
        { 17, 14, 12 },
        { 222, 34, 30 },
        { 72, 16, 44 },
        { 58, 32, 12 },
        { 10, 7, 6 },
    },
    .coef = {
        {
            {
                {
                    { { 195, 29, 183 }, { 84, 49, 136 }, { 8, 42, 71 } },
                    { { 31, 107, 169 }, { 35, 99, 159 }, { 17, 82, 140 },
                      { 8, 66, 114 }, { 2, 44, 76 }, { 1, 19, 32 } },
                    { { 40, 132, 201 }, { 29, 114, 187 }, { 13, 91, 157 },
                      { 7, 75, 127 }, { 3, 58, 95 }, { 1, 28, 47 } },
                    { { 69, 142, 221 }, { 42, 122, 201 }, { 15, 91, 159 },
                      { 6, 67, 121 }, { 1, 42, 77 }, { 1, 17, 31 } },
                    { { 102, 148, 228 }, { 67, 117, 204 }, { 17, 82, 154 },
                      { 6, 59, 114 }, { 2, 39, 75 }, { 1, 15, 29 } },
                    { { 156, 57, 233 }, { 119, 57, 212 }, { 58, 48, 163 },
                      { 29, 40, 124 }, { 12, 30, 81 }, { 3, 12, 31 } },
                },
                {
                    { { 191, 107, 226 }, { 124, 117, 204 }, { 25, 99, 155 } },
                    { { 29, 148, 210 }, { 37, 126, 194 }, { 8, 93, 157 },
                      { 2, 68, 118 }, { 1, 39, 69 }, { 1, 17, 33 } },
                    { { 41, 151, 213 }, { 27, 123, 193 }, { 3, 82, 144 },
                      { 1, 58, 105 }, { 1, 32, 60 }, { 1, 13, 26 } },
                    { { 59, 159, 220 }, { 23, 126, 198 }, { 4, 88, 151 },
                      { 1, 66, 114 }, { 1, 38, 71 }, { 1, 18, 34 } },
                    { { 114, 136, 232 }, { 51, 114, 207 }, { 11, 83, 155 },
                      { 3, 56, 105 }, { 1, 33, 65 }, { 1, 17, 34 } },
                    { { 149, 65, 234 }, { 121, 57, 215 }, { 61, 49, 166 },
                      { 28, 36, 114 }, { 12, 25, 76 }, { 3, 16, 42 } },
                },
            },
            {
                {
                    { { 214, 49, 220 }, { 132, 63, 188 }, { 42, 65, 137 } },
                    { { 85, 137, 221 }, { 104, 131, 216 }, { 49, 111, 192 },
                      { 21, 87, 155 }, { 2, 49, 87 }, { 1, 16, 28 } },
                    { { 89, 163, 230 }, { 90, 137, 220 }, { 29, 100, 183 },
                      { 10, 70, 135 }, { 2, 42, 81 }, { 1, 17, 33 } },
                    { { 108, 167, 237 }, { 55, 133, 222 }, { 15, 97, 179 },
                      { 4, 72, 135 }, { 1, 45, 85 }, { 1, 19, 38 } },
                    { { 124, 146, 240 }, { 66, 124, 224 }, { 17, 88, 175 },
                      { 4, 58, 122 }, { 1, 36, 75 }, { 1, 18, 37 } },
                    { { 141, 79, 241 }, { 126, 70, 227 }, { 66, 58, 182 },
                      { 30, 44, 136 }, { 12, 34, 96 }, { 2, 20, 47 } },
                },
                {
                    { { 229, 99, 249 }, { 143, 111, 235 }, { 46, 109, 192 } },
                    { { 82, 158, 236 }, { 94, 146, 224 }, { 25, 117, 191 },
                      { 9, 87, 149 }, { 3, 56, 99 }, { 1, 33, 57 } },
                    { { 83, 167, 237 }, { 68, 145, 222 }, { 10, 103, 177 },
                      { 2, 72, 131 }, { 1, 41, 79 }, { 1, 20, 39 } },
                    { { 99, 167, 239 }, { 47, 141, 224 }, { 10, 104, 178 },
                      { 2, 73, 133 }, { 1, 44, 85 }, { 1, 22, 47 } },
                    { { 127, 145, 243 }, { 71, 129, 228 }, { 17, 93, 177 },
                      { 3, 61, 124 }, { 1, 41, 84 }, { 1, 21, 52 } },
                    { { 157, 78, 244 }, { 140, 72, 231 }, { 69, 58, 184 },
                      { 31, 44, 137 }, { 14, 38, 105 }, { 8, 23, 61 } },
                },
            },
        },
        {
            {
                {
                    { { 125, 34, 187 }, { 52, 41, 133 }, { 6, 31, 56 } },
                    { { 37, 109, 153 }, { 51, 102, 147 }, { 23, 87, 128 },
                      { 8, 67, 101 }, { 1, 41, 63 }, { 1, 19, 29 } },
                    { { 31, 154, 185 }, { 17, 127, 175 }, { 6, 96, 145 },
                      { 2, 73, 114 }, { 1, 51, 82 }, { 1, 28, 45 } },
                    { { 23, 163, 200 }, { 10, 131, 185 }, { 2, 93, 148 },
                      { 1, 67, 111 }, { 1, 41, 69 }, { 1, 14, 24 } },
                    { { 29, 176, 217 }, { 12, 145, 201 }, { 3, 101, 156 },
                      { 1, 69, 111 }, { 1, 39, 63 }, { 1, 14, 23 } },
                    { { 57, 192, 233 }, { 25, 154, 215 }, { 6, 109, 167 },
                      { 3, 78, 118 }, { 1, 48, 69 }, { 1, 21, 29 } },
                },
                {
                    { { 202, 105, 245 }, { 108, 106, 216 }, { 18, 90, 144 } },
                    { { 33, 172, 219 }, { 64, 149, 206 }, { 14, 117, 177 },
                      { 5, 90, 141 }, { 2, 61, 95 }, { 1, 37, 57 } },
                    { { 33, 179, 220 }, { 11, 140, 198 }, { 1, 89, 148 },
                      { 1, 60, 104 }, { 1, 33, 57 }, { 1, 12, 21 } },
                    { { 30, 181, 221 }, { 8, 141, 198 }, { 1, 87, 145 },
                      { 1, 58, 100 }, { 1, 31, 55 }, { 1, 12, 20 } },
                    { { 32, 186, 224 }, { 7, 142, 198 }, { 1, 86, 143 },
                      { 1, 58, 100 }, { 1, 31, 55 }, { 1, 12, 22 } },
                    { { 57, 192, 227 }, { 20, 143, 204 }, { 3, 96, 154 },
                      { 1, 68, 112 }, { 1, 42, 69 }, { 1, 19, 32 } },
                },
            },
            {
                {
                    { { 212, 35, 215 }, { 113, 47, 169 }, { 29, 48, 105 } },
                    { { 74, 129, 203 }, { 106, 120, 203 }, { 49, 107, 178 },
                      { 19, 84, 144 }, { 4, 50, 84 }, { 1, 15, 25 } },
                    { { 71, 172, 217 }, { 44, 141, 209 }, { 15, 102, 173 },
                      { 6, 76, 133 }, { 2, 51, 89 }, { 1, 24, 42 } },
                    { { 64, 185, 231 }, { 31, 148, 216 }, { 8, 103, 175 },
                      { 3, 74, 131 }, { 1, 46, 81 }, { 1, 18, 30 } },
                    { { 65, 196, 235 }, { 25, 157, 221 }, { 5, 105, 174 },
                      { 1, 67, 120 }, { 1, 38, 69 }, { 1, 15, 30 } },
                    { { 65, 204, 238 }, { 30, 156, 224 }, { 7, 107, 177 },
                      { 2, 70, 124 }, { 1, 42, 73 }, { 1, 18, 34 } },
                },
                {
                    { { 225, 86, 251 }, { 144, 104, 235 }, { 42, 99, 181 } },
                    { { 85, 175, 239 }, { 112, 165, 229 }, { 29, 136, 200 },
                      { 12, 103, 162 }, { 6, 77, 123 }, { 2, 53, 84 } },
                    { { 75, 183, 239 }, { 30, 155, 221 }, { 3, 106, 171 },
                      { 1, 74, 128 }, { 1, 44, 76 }, { 1, 17, 28 } },
                    { { 73, 185, 240 }, { 27, 159, 222 }, { 2, 107, 172 },
                      { 1, 75, 127 }, { 1, 42, 73 }, { 1, 17, 29 } },
                    { { 62, 190, 238 }, { 21, 159, 222 }, { 2, 107, 172 },
                      { 1, 72, 122 }, { 1, 40, 71 }, { 1, 18, 32 } },
                    { { 61, 199, 240 }, { 27, 161, 226 }, { 4, 113, 180 },
                      { 1, 76, 129 }, { 1, 46, 80 }, { 1, 23, 41 } },
                },
            },
        },
        {
            {
                {
                    { { 7, 27, 153 }, { 5, 30, 95 }, { 1, 16, 30 } },
                    { { 50, 75, 127 }, { 57, 75, 124 }, { 27, 67, 108 },
                      { 10, 54, 86 }, { 1, 33, 52 }, { 1, 12, 18 } },
                    { { 43, 125, 151 }, { 26, 108, 148 }, { 7, 83, 122 },
                      { 2, 59, 89 }, { 1, 38, 60 }, { 1, 17, 27 } },
                    { { 23, 144, 163 }, { 13, 112, 154 }, { 2, 75, 117 },
                      { 1, 50, 81 }, { 1, 31, 51 }, { 1, 14, 23 } },
                    { { 18, 162, 185 }, { 6, 123, 171 }, { 1, 78, 125 },
                      { 1, 51, 86 }, { 1, 31, 54 }, { 1, 14, 23 } },
                    { { 15, 199, 227 }, { 3, 150, 204 }, { 1, 91, 146 },
                      { 1, 55, 95 }, { 1, 30, 53 }, { 1, 11, 20 } },
                },
                {
                    { { 19, 55, 240 }, { 19, 59, 196 }, { 3, 52, 105 } },
                    { { 41, 166, 207 }, { 104, 153, 199 }, { 31, 123, 181 },
                      { 14, 101, 152 }, { 5, 72, 106 }, { 1, 36, 52 } },
                    { { 35, 176, 211 }, { 12, 131, 190 }, { 2, 88, 144 },
                      { 1, 60, 101 }, { 1, 36, 60 }, { 1, 16, 28 } },
                    { { 28, 183, 213 }, { 8, 134, 191 }, { 1, 86, 142 },
                      { 1, 56, 96 }, { 1, 30, 53 }, { 1, 12, 20 } },
                    { { 20, 190, 215 }, { 4, 135, 192 }, { 1, 84, 139 },
                      { 1, 53, 91 }, { 1, 28, 49 }, { 1, 11, 20 } },
                    { { 13, 196, 216 }, { 2, 137, 192 }, { 1, 86, 143 },
                      { 1, 57, 99 }, { 1, 32, 56 }, { 1, 13, 24 } },
                },
            },
            {
                {
                    { { 211, 29, 217 }, { 96, 47, 156 }, { 22, 43, 87 } },
                    { { 78, 120, 193 }, { 111, 116, 186 }, { 46, 102, 164 },
                      { 15, 80, 128 }, { 2, 49, 76 }, { 1, 18, 28 } },
                    { { 71, 161, 203 }, { 42, 132, 192 }, { 10, 98, 150 },
                      { 3, 69, 109 }, { 1, 44, 70 }, { 1, 18, 29 } },
                    { { 57, 186, 211 }, { 30, 140, 196 }, { 4, 93, 146 },
                      { 1, 62, 102 }, { 1, 38, 65 }, { 1, 16, 27 } },
                    { { 47, 199, 217 }, { 14, 145, 196 }, { 1, 88, 142 },
                      { 1, 57, 98 }, { 1, 36, 62 }, { 1, 15, 26 } },
                    { { 26, 219, 229 }, { 5, 155, 207 }, { 1, 94, 151 },
                      { 1, 60, 104 }, { 1, 36, 62 }, { 1, 16, 28 } },
                },
                {
                    { { 233, 29, 248 }, { 146, 47, 220 }, { 43, 52, 140 } },
                    { { 100, 163, 232 }, { 179, 161, 222 }, { 63, 142, 204 },
                      { 37, 113, 174 }, { 26, 89, 137 }, { 18, 68, 97 } },
                    { { 85, 181, 230 }, { 32, 146, 209 }, { 7, 100, 164 },
                      { 3, 71, 121 }, { 1, 45, 77 }, { 1, 18, 30 } },
                    { { 65, 187, 230 }, { 20, 148, 207 }, { 2, 97, 159 },
                      { 1, 68, 116 }, { 1, 40, 70 }, { 1, 14, 29 } },
                    { { 40, 194, 227 }, { 8, 147, 204 }, { 1, 94, 155 },
                      { 1, 65, 112 }, { 1, 39, 66 }, { 1, 14, 26 } },
                    { { 16, 208, 228 }, { 3, 151, 207 }, { 1, 98, 160 },
                      { 1, 67, 117 }, { 1, 41, 74 }, { 1, 17, 31 } },
                },
            },
        },
        {
            {
                {
                    { { 17, 38, 140 }, { 7, 34, 80 }, { 1, 17, 29 } },
                    { { 37, 75, 128 }, { 41, 76, 128 }, { 26, 66, 116 },
                      { 12, 52, 94 }, { 2, 32, 55 }, { 1, 10, 16 } },
                    { { 50, 127, 154 }, { 37, 109, 152 }, { 16, 82, 121 },
                      { 5, 59, 85 }, { 1, 35, 54 }, { 1, 13, 20 } },
                    { { 40, 142, 167 }, { 17, 110, 157 }, { 2, 71, 112 },
                      { 1, 44, 72 }, { 1, 27, 45 }, { 1, 11, 17 } },
                    { { 30, 175, 188 }, { 9, 124, 169 }, { 1, 74, 116 },
                      { 1, 48, 78 }, { 1, 30, 49 }, { 1, 11, 18 } },
                    { { 10, 222, 223 }, { 2, 150, 194 }, { 1, 83, 128 },
                      { 1, 48, 79 }, { 1, 27, 45 }, { 1, 11, 17 } },
                },
                {
                    { { 36, 41, 235 }, { 29, 36, 193 }, { 10, 27, 111 } },
                    { { 85, 165, 222 }, { 177, 162, 215 }, { 110, 135, 195 },
                      { 57, 113, 168 }, { 23, 83, 120 }, { 10, 49, 61 } },
                    { { 85, 190, 223 }, { 36, 139, 200 }, { 5, 90, 146 },
                      { 1, 60, 103 }, { 1, 38, 65 }, { 1, 18, 30 } },
                    { { 72, 202, 223 }, { 23, 141, 199 }, { 2, 86, 140 },
                      { 1, 56, 97 }, { 1, 36, 61 }, { 1, 16, 27 } },
                    { { 55, 218, 225 }, { 13, 145, 200 }, { 1, 86, 141 },
                      { 1, 57, 99 }, { 1, 35, 61 }, { 1, 13, 22 } },
                    { { 15, 235, 212 }, { 1, 132, 184 }, { 1, 84, 139 },
                      { 1, 57, 97 }, { 1, 34, 56 }, { 1, 14, 23 } },
                },
            },
            {
                {
                    { { 181, 21, 201 }, { 61, 37, 123 }, { 10, 38, 71 } },
                    { { 47, 106, 172 }, { 95, 104, 173 }, { 42, 93, 159 },
                      { 18, 77, 131 }, { 4, 50, 81 }, { 1, 17, 23 } },
                    { { 62, 147, 199 }, { 44, 130, 189 }, { 28, 102, 154 },
                      { 18, 75, 115 }, { 2, 44, 65 }, { 1, 12, 19 } },
                    { { 55, 153, 210 }, { 24, 130, 194 }, { 3, 93, 146 },
                      { 1, 61, 97 }, { 1, 31, 50 }, { 1, 10, 16 } },
                    { { 49, 186, 223 }, { 17, 148, 204 }, { 1, 96, 142 },
                      { 1, 53, 83 }, { 1, 26, 44 }, { 1, 11, 17 } },
                    { { 13, 217, 212 }, { 2, 136, 180 }, { 1, 78, 124 },
                      { 1, 50, 83 }, { 1, 29, 49 }, { 1, 14, 23 } },
                },
                {
                    { { 197, 13, 247 }, { 82, 17, 222 }, { 25, 17, 162 } },
                    { { 126, 186, 247 }, { 234, 191, 243 }, { 176, 177, 234 },
                      { 104, 158, 220 }, { 66, 128, 186 }, { 55, 90, 137 } },
                    { { 111, 197, 242 }, { 46, 158, 219 }, { 9, 104, 171 },
                      { 2, 65, 125 }, { 1, 44, 80 }, { 1, 17, 91 } },
                    { { 104, 208, 245 }, { 39, 168, 224 }, { 3, 109, 162 },
                      { 1, 79, 124 }, { 1, 50, 102 }, { 1, 43, 102 } },
                    { { 84, 220, 246 }, { 31, 177, 231 }, { 2, 115, 180 },
                      { 1, 79, 134 }, { 1, 55, 77 }, { 1, 60, 79 } },
                    { { 43, 243, 240 }, { 8, 180, 217 }, { 1, 115, 166 },
                      { 1, 84, 121 }, { 1, 51, 67 }, { 1, 16, 6 } },
                },
            },
        },
    },
    .switchable_interp = { { 235, 162 }, { 36, 255 }, { 34, 3 }, { 149, 144 } },
    .inter_mode = {
        { 2, 173, 34 },
        { 7, 145, 85 },
        { 7, 166, 63 },
        { 7, 94, 66 },
        { 8, 64, 46 },
        { 17, 81, 31 },
        { 25, 29, 30 },
    },
    .intra_inter = { 9, 102, 187, 225 },
    .comp_inter = { 239, 183, 119, 96, 41 },
    .single_ref = { { 33, 16 }, { 77, 74 }, { 142, 142 }, { 172, 170 }, { 238, 247 } },
    .comp_ref = { 50, 126, 123, 221, 226 },
    .tx32 = { { 3, 136, 37 }, { 5, 52, 13 } },
    .tx16 = { { 20, 152 }, { 15, 101 } },
    .tx8 = { { 100 }, { 66 } },
    .skip = { 192, 128, 64 },
    .mv = {
        .joints = { 32, 64, 96 },
        .comps = {
            {
                .sign = 128,
                .classes = { 224, 144, 192, 168, 192, 176, 192, 198, 198, 245 },
                .class0 = { 216 },
                .bits = { 136, 140, 148, 160, 176, 192, 224, 234, 234, 240 },
                .class0_fp = { { 128, 128, 64 }, { 96, 112, 64 } },
                .fp = { 64, 96, 64 },
                .class0_hp = 160,
                .hp = 128,
            },
            {
                .sign = 128,
                .classes = { 216, 128, 176, 160, 176, 176, 192, 198, 198, 208 },
                .class0 = { 208 },
                .bits = { 136, 140, 148, 160, 176, 192, 224, 234, 234, 240 },
                .class0_fp = { { 128, 128, 64 }, { 96, 112, 64 } },
                .fp = { 64, 96, 64 },
                .class0_hp = 160,
                .hp = 128,
            },
        },
    },
};

// vp9_kf_y_mode_prob, vp9_kf_uv_mode_prob, vp9_kf_partition_probs: intra frames' probs by above/left context
static const uint8_t kf_y_mode[10][10][9] = {
    {
        { 137, 30, 42, 148, 151, 207, 70, 52, 91 },
        { 92, 45, 102, 136, 116, 180, 74, 90, 100 },
        { 73, 32, 19, 187, 222, 215, 46, 34, 100 },
        { 91, 30, 32, 116, 121, 186, 93, 86, 94 },
        { 72, 35, 36, 149, 68, 206, 68, 63, 105 },
        { 73, 31, 28, 138, 57, 124, 55, 122, 151 },
        { 67, 23, 21, 140, 126, 197, 40, 37, 171 },
        { 86, 27, 28, 128, 154, 212, 45, 43, 53 },
        { 74, 32, 27, 107, 86, 160, 63, 134, 102 },
        { 59, 67, 44, 140, 161, 202, 78, 67, 119 },
    },
    {
        { 63, 36, 126, 146, 123, 158, 60, 90, 96 },
        { 43, 46, 168, 134, 107, 128, 69, 142, 92 },
        { 44, 29, 68, 159, 201, 177, 50, 57, 77 },
        { 58, 38, 76, 114, 97, 172, 78, 133, 92 },
        { 46, 41, 76, 140, 63, 184, 69, 112, 57 },
        { 38, 32, 85, 140, 46, 112, 54, 151, 133 },
        { 39, 27, 61, 131, 110, 175, 44, 75, 136 },
        { 52, 30, 74, 113, 130, 175, 51, 64, 58 },
        { 47, 35, 80, 100, 74, 143, 64, 163, 74 },
        { 36, 61, 116, 114, 128, 162, 80, 125, 82 },
    },
    {
        { 82, 26, 26, 171, 208, 204, 44, 32, 105 },
        { 55, 44, 68, 166, 179, 192, 57, 57, 108 },
        { 42, 26, 11, 199, 241, 228, 23, 15, 85 },
        { 68, 42, 19, 131, 160, 199, 55, 52, 83 },
        { 58, 50, 25, 139, 115, 232, 39, 52, 118 },
        { 50, 35, 33, 153, 104, 162, 64, 59, 131 },
        { 44, 24, 16, 150, 177, 202, 33, 19, 156 },
        { 55, 27, 12, 153, 203, 218, 26, 27, 49 },
        { 53, 49, 21, 110, 116, 168, 59, 80, 76 },
        { 38, 72, 19, 168, 203, 212, 50, 50, 107 },
    },
    {
        { 103, 26, 36, 129, 132, 201, 83, 80, 93 },
        { 59, 38, 83, 112, 103, 162, 98, 136, 90 },
        { 62, 30, 23, 158, 200, 207, 59, 57, 50 },
        { 67, 30, 29, 84, 86, 191, 102, 91, 59 },
        { 60, 32, 33, 112, 71, 220, 64, 89, 104 },
        { 53, 26, 34, 130, 56, 149, 84, 120, 103 },
        { 53, 21, 23, 133, 109, 210, 56, 77, 172 },
        { 77, 19, 29, 112, 142, 228, 55, 66, 36 },
        { 61, 29, 29, 93, 97, 165, 83, 175, 162 },
        { 47, 47, 43, 114, 137, 181, 100, 99, 95 },
    },
    {
        { 69, 23, 29, 128, 83, 199, 46, 44, 101 },
        { 53, 40, 55, 139, 69, 183, 61, 80, 110 },
        { 40, 29, 19, 161, 180, 207, 43, 24, 91 },
        { 60, 34, 19, 105, 61, 198, 53, 64, 89 },
        { 52, 31, 22, 158, 40, 209, 58, 62, 89 },
        { 44, 31, 29, 147, 46, 158, 56, 102, 198 },
        { 35, 19, 12, 135, 87, 209, 41, 45, 167 },
        { 55, 25, 21, 118, 95, 215, 38, 39, 66 },
        { 51, 38, 25, 113, 58, 164, 70, 93, 97 },
        { 47, 54, 34, 146, 108, 203, 72, 103, 151 },
    },
    {
        { 64, 19, 37, 156, 66, 138, 49, 95, 133 },
        { 46, 27, 80, 150, 55, 124, 55, 121, 135 },
        { 36, 23, 27, 165, 149, 166, 54, 64, 118 },
        { 53, 21, 36, 131, 63, 163, 60, 109, 81 },
        { 40, 26, 35, 154, 40, 185, 51, 97, 123 },
        { 35, 19, 34, 179, 19, 97, 48, 129, 124 },
        { 36, 20, 26, 136, 62, 164, 33, 77, 154 },
        { 45, 18, 32, 130, 90, 157, 40, 79, 91 },
        { 45, 26, 28, 129, 45, 129, 49, 147, 123 },
        { 38, 44, 51, 136, 74, 162, 57, 97, 121 },
    },
    {
        { 75, 17, 22, 136, 138, 185, 32, 34, 166 },
        { 56, 39, 58, 133, 117, 173, 48, 53, 187 },
        { 35, 21, 12, 161, 212, 207, 20, 23, 145 },
        { 56, 29, 19, 117, 109, 181, 55, 68, 112 },
        { 47, 29, 17, 153, 64, 220, 59, 51, 114 },
        { 46, 16, 24, 136, 76, 147, 41, 64, 172 },
        { 34, 17, 11, 108, 152, 187, 13, 15, 209 },
        { 51, 24, 14, 115, 133, 209, 32, 26, 104 },
        { 55, 30, 18, 122, 79, 179, 44, 88, 116 },
        { 37, 49, 25, 129, 168, 164, 41, 54, 148 },
    },
    {
        { 82, 22, 32, 127, 143, 213, 39, 41, 70 },
        { 62, 44, 61, 123, 105, 189, 48, 57, 64 },
        { 47, 25, 17, 175, 222, 220, 24, 30, 86 },
        { 68, 36, 17, 106, 102, 206, 59, 74, 74 },
        { 57, 39, 23, 151, 68, 216, 55, 63, 58 },
        { 49, 30, 35, 141, 70, 168, 82, 40, 115 },
        { 51, 25, 15, 136, 129, 202, 38, 35, 139 },
        { 68, 26, 16, 111, 141, 215, 29, 28, 28 },
        { 59, 39, 19, 114, 75, 180, 77, 104, 42 },
        { 40, 61, 26, 126, 152, 206, 61, 59, 93 },
    },
    {
        { 78, 23, 39, 111, 117, 170, 74, 124, 94 },
        { 48, 34, 86, 101, 92, 146, 78, 179, 134 },
        { 47, 22, 24, 138, 187, 178, 68, 69, 59 },
        { 56, 25, 33, 105, 112, 187, 95, 177, 129 },
        { 48, 31, 27, 114, 63, 183, 82, 116, 56 },
        { 43, 28, 37, 121, 63, 123, 61, 192, 169 },
        { 42, 17, 24, 109, 97, 177, 56, 76, 122 },
        { 58, 18, 28, 105, 139, 182, 70, 92, 63 },
        { 46, 23, 32, 74, 86, 150, 67, 183, 88 },
        { 36, 38, 48, 92, 122, 165, 88, 137, 91 },
    },
    {
        { 65, 70, 60, 155, 159, 199, 61, 60, 81 },
        { 44, 78, 115, 132, 119, 173, 71, 112, 93 },
        { 39, 38, 21, 184, 227, 206, 42, 32, 64 },
        { 58, 47, 36, 124, 137, 193, 80, 82, 78 },
        { 49, 50, 35, 144, 95, 205, 63, 78, 59 },
        { 41, 53, 52, 148, 71, 142, 65, 128, 51 },
        { 40, 36, 28, 143, 143, 202, 40, 55, 137 },
        { 52, 34, 29, 129, 183, 227, 42, 35, 43 },
        { 42, 44, 44, 104, 105, 164, 64, 130, 80 },
        { 43, 81, 53, 140, 169, 204, 68, 84, 72 },
    },
};

static const uint8_t kf_uv_mode[10][9] = {
    { 144, 11, 54, 157, 195, 130, 46, 58, 108 },
    { 118, 15, 123, 148, 131, 101, 44, 93, 131 },
    { 113, 12, 23, 188, 226, 142, 26, 32, 125 },
    { 120, 11, 50, 123, 163, 135, 64, 77, 103 },
    { 113, 9, 36, 155, 111, 157, 32, 44, 161 },
    { 116, 9, 55, 176, 76, 96, 37, 61, 149 },
    { 115, 9, 28, 141, 161, 167, 21, 25, 193 },
    { 120, 12, 32, 145, 195, 142, 32, 38, 86 },
    { 116, 12, 64, 120, 140, 125, 49, 115, 121 },
    { 102, 19, 66, 162, 182, 122, 35, 59, 128 },
};

static const uint8_t kf_partition[16][3] = {
    { 158, 97, 94 },
    { 93, 24, 99 },
    { 85, 119, 44 },
    { 62, 59, 67 },
    { 149, 53, 53 },
    { 94, 20, 48 },
    { 83, 53, 24 },
    { 52, 18, 18 },
    { 150, 40, 39 },
    { 78, 12, 26 },
    { 67, 33, 11 },
    { 24, 7, 5 },
    { 174, 35, 49 },
    { 68, 11, 27 },
    { 57, 15, 9 },
    { 12, 3, 3 },
};

static const uint8_t inv_map_table[255] = {
    7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228, 241, 254,
    1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87,
    88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
    109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 151, 152,
    153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
    174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195,
    196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 216, 217,
    218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
    239, 240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 253,
};

// libvpx's trees, leaves as minus the symbol
static const int8_t intra_mode_tree[18] = { 0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7 };
static const int8_t inter_mode_tree[6] = { -2, 2, 0, 4, -1, -3 };  // NEARESTMV, NEARMV, ZEROMV, NEWMV from 0
static const int8_t switchable_interp_tree[4] = { 0, 2, -1, -2 };
static const int8_t four_tree[6] = { 0, 2, -1, 4, -2, -3 };  // partitions, MV joints and fractions, 32x32 transforms
static const int8_t tx16_tree[4] = { 0, 2, -1, -2 };
static const int8_t binary_tree[2] = { 0, -1 };
static const int8_t mv_class_tree[20] = { 0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10 };

// nvdec_vp9EntropyProbs_t: 9-prob rows split 8 + tail, 3-prob rows padded to 4, key frame partitions beside context's
struct nvdec_probs {
    uint8_t kf_y_mode[10][10][8], kf_y_mode_tail[10][10];
    uint8_t ref_pred[3], seg_tree[7], seg_pred[3], ref_scores[4], comp_pred[2], pad0[9];
    uint8_t kf_uv_mode[10][8], kf_uv_mode_tail[10], pad1[6];
    uint8_t inter_mode[7][4], intra_inter[4];
    uint8_t uv_mode[10][8], tx8[2][1], tx16[2][2], tx32[2][3], y_mode_tail[4], y_mode[4][8];
    uint8_t partition[2][16][4];
    uint8_t uv_mode_tail[10], switchable_interp[4][2], comp_inter[5], skip[3], pad2;
    uint8_t mv_joints[3], mv_sign[2], mv_class0[2][1], mv_fp[2][3], mv_class0_hp[2], mv_hp[2], mv_classes[2][10];
    uint8_t mv_class0_fp[2][2][3], mv_bits[2][10];
    uint8_t single_ref[5][2], comp_ref[5], pad3[17];
    uint8_t coef[4][2][2][6][6][4];
};

_Static_assert(sizeof(struct nvdec_probs) == VP9_PROBS_SIZE, "nvdec_vp9EntropyProbs_t is 3744 bytes");

void vp9_default_probs(struct vp9_probs *p)
{
    *p = defaults;
}

static int inv_recenter_nonneg(int v, int m)
{
    if (v > 2 * m)
        return v;
    return v & 1 ? m - ((v + 1) >> 1) : m + (v >> 1);
}

static int inv_remap_prob(int v, int m)
{
    v = inv_map_table[v];
    m--;
    if (2 * m <= 255)
        return 1 + inv_recenter_nonneg(v, m);
    return 255 - inv_recenter_nonneg(v, 254 - m);
}

static int decode_term_subexp(struct boolreader *r)
{
    if (!bool_literal(r, 1))
        return (int)bool_literal(r, 4);
    if (!bool_literal(r, 1))
        return (int)bool_literal(r, 4) + 16;
    if (!bool_literal(r, 1))
        return (int)bool_literal(r, 5) + 32;
    int v = (int)bool_literal(r, 7);
    return v < 65 ? v + 64 : (v << 1) - 1 + (int)bool_literal(r, 1);
}

static void diff_update(struct boolreader *r, uint8_t *p)
{
    if (bool_read(r, 252))
        *p = (uint8_t)inv_remap_prob(decode_term_subexp(r), *p);
}

static void diff_update_n(struct boolreader *r, uint8_t *p, int n)
{
    for (int i = 0; i < n; i++)
        diff_update(r, &p[i]);
}

static void mv_update_n(struct boolreader *r, uint8_t *p, int n)
{
    for (int i = 0; i < n; i++)
        if (bool_read(r, 252))
            p[i] = (uint8_t)(bool_literal(r, 7) << 1 | 1);
}

int vp9_compressed(struct vp9_prob_frame *f, const uint8_t *data, size_t len, struct vp9_probs *p)
{
    static const uint8_t biggest_tx[5] = { 0, 1, 2, 3, 3 };
    struct boolreader r;
    bool_init(&r, data, len);
    if (!len || bool_read(&r, 128))  // the marker bit
        return -1;
    f->tx_mode = VP9_ONLY_4X4;
    if (!f->lossless) {
        f->tx_mode = bool_literal(&r, 2);
        if (f->tx_mode == VP9_ALLOW_32X32)
            f->tx_mode += bool_literal(&r, 1);
    }
    if (f->tx_mode == VP9_TX_MODE_SELECT) {
        for (int i = 0; i < 2; i++)
            diff_update_n(&r, p->tx8[i], 1);
        for (int i = 0; i < 2; i++)
            diff_update_n(&r, p->tx16[i], 2);
        for (int i = 0; i < 2; i++)
            diff_update_n(&r, p->tx32[i], 3);
    }
    for (uint32_t t = 0; t <= biggest_tx[f->tx_mode]; t++) {
        if (!bool_literal(&r, 1))
            continue;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 6; k++)
                    for (int l = 0; l < (k ? 6 : 3); l++)
                        diff_update_n(&r, p->coef[t][i][j][k][l], 3);
    }
    diff_update_n(&r, p->skip, 3);
    f->reference_mode = VP9_SINGLE_REFERENCE;
    if (f->intra)
        return 0;

    for (int i = 0; i < 7; i++)
        diff_update_n(&r, p->inter_mode[i], 3);
    for (int i = 0; f->switchable && i < 4; i++)
        diff_update_n(&r, p->switchable_interp[i], 2);
    diff_update_n(&r, p->intra_inter, 4);
    if (f->compound && bool_literal(&r, 1))
        f->reference_mode = bool_literal(&r, 1) ? VP9_REFERENCE_MODE_SELECT : VP9_COMPOUND_REFERENCE;
    if (f->reference_mode == VP9_REFERENCE_MODE_SELECT)
        diff_update_n(&r, p->comp_inter, 5);
    for (int i = 0; f->reference_mode != VP9_COMPOUND_REFERENCE && i < 5; i++)
        diff_update_n(&r, p->single_ref[i], 2);
    if (f->reference_mode != VP9_SINGLE_REFERENCE)
        diff_update_n(&r, p->comp_ref, 5);
    for (int i = 0; i < 4; i++)
        diff_update_n(&r, p->y_mode[i], 9);
    for (int i = 0; i < 16; i++)
        diff_update_n(&r, p->partition[i], 3);

    mv_update_n(&r, p->mv.joints, 3);
    for (int i = 0; i < 2; i++) {
        struct vp9_mv_comp *c = &p->mv.comps[i];
        mv_update_n(&r, &c->sign, 1);
        mv_update_n(&r, c->classes, 10);
        mv_update_n(&r, c->class0, 1);
        mv_update_n(&r, c->bits, 10);
    }
    for (int i = 0; i < 2; i++) {
        struct vp9_mv_comp *c = &p->mv.comps[i];
        for (int j = 0; j < 2; j++)
            mv_update_n(&r, c->class0_fp[j], 3);
        mv_update_n(&r, c->fp, 3);
    }
    for (int i = 0; f->allow_hp && i < 2; i++) {
        mv_update_n(&r, &p->mv.comps[i].class0_hp, 1);
        mv_update_n(&r, &p->mv.comps[i].hp, 1);
    }
    return 0;
}

static uint8_t get_prob(uint32_t num, uint32_t den)
{
    uint64_t p = ((uint64_t)num * 256 + (den >> 1)) / den;
    return (uint8_t)(p > 255 ? 255 : p < 1 ? 1 : p);
}

// merge_probs(), and mode_mv_merge_probs() with a count_sat of 20 and a max_update_factor of 128
static uint8_t merge(uint8_t pre, uint32_t ct0, uint32_t ct1, uint32_t count_sat, uint32_t max_factor)
{
    uint32_t den = ct0 + ct1, count = den < count_sat ? den : count_sat;
    if (!den)
        return pre;
    uint32_t factor = max_factor * count / count_sat;
    return (uint8_t)((pre * (256 - factor) + get_prob(ct0, den) * factor + 128) >> 8);
}

static uint8_t merge_mode(uint8_t pre, const uint32_t *ct)
{
    return merge(pre, ct[0], ct[1], 20, 128);
}

static uint32_t tree_merge(const int8_t *tree, int i, const uint8_t *pre, const uint32_t *counts, uint8_t *probs)
{
    int l = tree[i], r = tree[i + 1];
    uint32_t left = l <= 0 ? counts[-l] : tree_merge(tree, l, pre, counts, probs);
    uint32_t right = r <= 0 ? counts[-r] : tree_merge(tree, r, pre, counts, probs);
    probs[i >> 1] = merge(pre[i >> 1], left, right, 20, 128);
    return left + right;
}

static void merge_mv(int allow_hp, const struct vp9_probs *pre, const struct vp9_counts *c, struct vp9_probs *p)
{
    tree_merge(four_tree, 0, pre->mv.joints, c->mv_joints, p->mv.joints);
    for (int i = 0; i < 2; i++) {
        const struct vp9_mv_comp *from = &pre->mv.comps[i];
        struct vp9_mv_comp *to = &p->mv.comps[i];
        to->sign = merge_mode(from->sign, c->mv_sign[i]);
        tree_merge(mv_class_tree, 0, from->classes, c->mv_classes[i], to->classes);
        tree_merge(binary_tree, 0, from->class0, c->mv_class0[i], to->class0);
        for (int j = 0; j < 10; j++)
            to->bits[j] = merge_mode(from->bits[j], c->mv_bits[i][j]);
        for (int j = 0; j < 2; j++)
            tree_merge(four_tree, 0, from->class0_fp[j], c->mv_class0_fp[i][j], to->class0_fp[j]);
        tree_merge(four_tree, 0, from->fp, c->mv_fp[i], to->fp);
        if (allow_hp) {
            to->class0_hp = merge_mode(from->class0_hp, c->mv_class0_hp[i]);
            to->hp = merge_mode(from->hp, c->mv_hp[i]);
        }
    }
}

// vp9_adapt_coef_probs(), then for inter frames vp9_adapt_mode_probs() and vp9_adapt_mv_probs()
void vp9_adapt(const struct vp9_prob_frame *f, const struct vp9_probs *pre, const struct vp9_counts *c,
               struct vp9_probs *p)
{
    uint32_t factor = !f->intra && f->after_key ? 128 : 112;
    for (int t = 0; t < 4; t++)
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 6; k++)
                    for (int l = 0; l < (k ? 6 : 3); l++) {
                        const uint32_t *n = c->coef[t][i][j][k][l];
                        const uint8_t *from = pre->coef[t][i][j][k][l];
                        uint8_t *to = p->coef[t][i][j][k][l];
                        to[0] = merge(from[0], n[3], c->eob_branch[t][i][j][k][l] - n[3], 24, factor);
                        to[1] = merge(from[1], n[0], n[1] + n[2], 24, factor);
                        to[2] = merge(from[2], n[1], n[2], 24, factor);
                    }
    if (f->intra)
        return;

    for (int i = 0; i < 4; i++)
        p->intra_inter[i] = merge_mode(pre->intra_inter[i], c->intra_inter[i]);
    for (int i = 0; i < 5; i++) {
        p->comp_inter[i] = merge_mode(pre->comp_inter[i], c->comp_inter[i]);
        p->comp_ref[i] = merge_mode(pre->comp_ref[i], c->comp_ref[i]);
        for (int j = 0; j < 2; j++)
            p->single_ref[i][j] = merge_mode(pre->single_ref[i][j], c->single_ref[i][j]);
    }
    for (int i = 0; i < 7; i++) {
        const uint32_t (*b)[2] = c->inter_mode[i];
        uint32_t modes[4] = { b[1][0], b[2][0], b[0][0], b[2][1] };
        tree_merge(inter_mode_tree, 0, pre->inter_mode[i], modes, p->inter_mode[i]);
    }
    for (int i = 0; i < 4; i++)
        tree_merge(intra_mode_tree, 0, pre->y_mode[i], c->y_mode[i], p->y_mode[i]);
    for (int i = 0; i < 10; i++)
        tree_merge(intra_mode_tree, 0, pre->uv_mode[i], c->uv_mode[i], p->uv_mode[i]);
    for (int i = 0; i < 16; i++)
        tree_merge(four_tree, 0, pre->partition[i], c->partition[i], p->partition[i]);
    for (int i = 0; f->switchable && i < 4; i++)
        tree_merge(switchable_interp_tree, 0, pre->switchable_interp[i], c->switchable_interp[i],
                   p->switchable_interp[i]);
    for (int i = 0; f->tx_mode == VP9_TX_MODE_SELECT && i < 2; i++) {
        tree_merge(binary_tree, 0, pre->tx8[i], c->tx8[i], p->tx8[i]);
        tree_merge(tx16_tree, 0, pre->tx16[i], c->tx16[i], p->tx16[i]);
        tree_merge(four_tree, 0, pre->tx32[i], c->tx32[i], p->tx32[i]);
    }
    for (int i = 0; i < 3; i++)
        p->skip[i] = merge_mode(pre->skip[i], c->skip[i]);
    merge_mv(f->allow_hp, pre, c, p);
}

static void split9(const uint8_t *probs, uint8_t *eight, uint8_t *tail)
{
    memcpy(eight, probs, 8);
    *tail = probs[8];
}

void vp9_pack(const struct vp9_probs *p, const uint8_t *seg_tree, const uint8_t *seg_pred, uint8_t *table)
{
    struct nvdec_probs *t = (struct nvdec_probs *)table;
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++)
            split9(kf_y_mode[i][j], t->kf_y_mode[i][j], &t->kf_y_mode_tail[i][j]);
        split9(kf_uv_mode[i], t->kf_uv_mode[i], &t->kf_uv_mode_tail[i]);
        split9(p->uv_mode[i], t->uv_mode[i], &t->uv_mode_tail[i]);
    }
    memcpy(t->seg_tree, seg_tree, 7);
    memcpy(t->seg_pred, seg_pred, 3);
    for (int i = 0; i < 7; i++)
        memcpy(t->inter_mode[i], p->inter_mode[i], 3);
    memcpy(t->intra_inter, p->intra_inter, 4);
    memcpy(t->tx8, p->tx8, sizeof(t->tx8));
    memcpy(t->tx16, p->tx16, sizeof(t->tx16));
    memcpy(t->tx32, p->tx32, sizeof(t->tx32));
    for (int i = 0; i < 4; i++)
        split9(p->y_mode[i], t->y_mode[i], &t->y_mode_tail[i]);
    for (int i = 0; i < 16; i++) {
        memcpy(t->partition[0][i], kf_partition[i], 3);
        memcpy(t->partition[1][i], p->partition[i], 3);
    }
    memcpy(t->switchable_interp, p->switchable_interp, sizeof(t->switchable_interp));
    memcpy(t->comp_inter, p->comp_inter, 5);
    memcpy(t->skip, p->skip, 3);
    memcpy(t->mv_joints, p->mv.joints, 3);
    for (int i = 0; i < 2; i++) {
        const struct vp9_mv_comp *c = &p->mv.comps[i];
        t->mv_sign[i] = c->sign;
        t->mv_class0[i][0] = c->class0[0];
        memcpy(t->mv_fp[i], c->fp, 3);
        t->mv_class0_hp[i] = c->class0_hp;
        t->mv_hp[i] = c->hp;
        memcpy(t->mv_classes[i], c->classes, 10);
        memcpy(t->mv_class0_fp[i], c->class0_fp, sizeof(t->mv_class0_fp[i]));
        memcpy(t->mv_bits[i], c->bits, 10);
    }
    memcpy(t->single_ref, p->single_ref, sizeof(t->single_ref));
    memcpy(t->comp_ref, p->comp_ref, 5);
    uint8_t (*to)[4] = (uint8_t (*)[4])t->coef;
    const uint8_t (*from)[3] = (const uint8_t (*)[3])p->coef;
    for (int i = 0; i < 4 * 2 * 2 * 6 * 6; i++)
        memcpy(to[i], from[i], 3);
}
