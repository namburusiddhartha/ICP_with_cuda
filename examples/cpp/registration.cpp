/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
**/
#include <lzf.h>

#include <cstdint>
#include <cstdio>
#include <sstream>

#include "cupoch/cupoch.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/io/class_io/pointcloud_io.h"
#include "cupoch/registration/registration.h"
#include "cupoch/registration/transformation_estimation.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/filesystem.h"
#include "cupoch/utility/helper.h"

// References for PCD file IO
// http://pointclouds.org/documentation/tutorials/pcd_file_format.php
// https://github.com/PointCloudLibrary/pcl/blob/master/io/src/pcd_io.cpp
// https://www.mathworks.com/matlabcentral/fileexchange/40382-matlab-to-point-cloud-library

namespace cupoch {

namespace {
using namespace io;

enum PCDDataType {
    PCD_DATA_ASCII = 0,
    PCD_DATA_BINARY = 1,
    PCD_DATA_BINARY_COMPRESSED = 2
};

struct PCLPointField {
public:
    std::string name;
    int size;
    char type;
    int count;
    // helper variable
    int count_offset;
    int offset;
};

struct PCDHeader {
public:
    std::string version;
    std::vector<PCLPointField> fields;
    int width;
    int height;
    int points;
    PCDDataType datatype;
    std::string viewpoint;
    // helper variables
    int elementnum;
    int pointsize;
    bool has_points;
    bool has_normals;
    bool has_colors;
};

bool CheckHeader(PCDHeader &header) {
    if (header.points <= 0 || header.pointsize <= 0) {
        utility::LogWarning("[CheckHeader] PCD has no data.\n");
        return false;
    }
    if (header.fields.size() == 0 || header.pointsize <= 0) {
        utility::LogWarning("[CheckHeader] PCD has no fields.\n");
        return false;
    }
    header.has_points = false;
    header.has_normals = false;
    header.has_colors = false;
    bool has_x = false;
    bool has_y = false;
    bool has_z = false;
    bool has_normal_x = false;
    bool has_normal_y = false;
    bool has_normal_z = false;
    bool has_rgb = false;
    bool has_rgba = false;
    for (const auto &field : header.fields) {
        if (field.name == "x") {
            has_x = true;
        } else if (field.name == "y") {
            has_y = true;
        } else if (field.name == "z") {
            has_z = true;
        } else if (field.name == "normal_x") {
            has_normal_x = true;
        } else if (field.name == "normal_y") {
            has_normal_y = true;
        } else if (field.name == "normal_z") {
            has_normal_z = true;
        } else if (field.name == "rgb") {
            has_rgb = true;
        } else if (field.name == "rgba") {
            has_rgba = true;
        }
    }
    header.has_points = (has_x && has_y && has_z);
    header.has_normals = (has_normal_x && has_normal_y && has_normal_z);
    header.has_colors = (has_rgb || has_rgba);
    if (header.has_points == false) {
        utility::LogWarning(
                "[CheckHeader] Fields for point data are not complete.\n");
        return false;
    }
    return true;
}

bool ReadPCDHeader(FILE *file, PCDHeader &header) {
    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    size_t specified_channel_count = 0;

    while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
        std::string line(line_buffer);
        if (line == "") {
            continue;
        }
        std::vector<std::string> st;
        utility::SplitString(st, line, "\t\r\n ");
        std::stringstream sstream(line);
        sstream.imbue(std::locale::classic());
        std::string line_type;
        sstream >> line_type;
        if (line_type.substr(0, 1) == "#") {
        } else if (line_type.substr(0, 7) == "VERSION") {
            if (st.size() >= 2) {
                header.version = st[1];
            }
        } else if (line_type.substr(0, 6) == "FIELDS" ||
                   line_type.substr(0, 7) == "COLUMNS") {
            specified_channel_count = st.size() - 1;
            if (specified_channel_count == 0) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.\n");
                return false;
            }
            header.fields.resize(specified_channel_count);
            int count_offset = 0, offset = 0;
            for (size_t i = 0; i < specified_channel_count;
                 i++, count_offset += 1, offset += 4) {
                header.fields[i].name = st[i + 1];
                header.fields[i].size = 4;
                header.fields[i].type = 'F';
                header.fields[i].count = 1;
                header.fields[i].count_offset = count_offset;
                header.fields[i].offset = offset;
            }
            header.elementnum = count_offset;
            header.pointsize = offset;
        } else if (line_type.substr(0, 4) == "SIZE") {
            if (specified_channel_count != st.size() - 1) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.\n");
                return false;
            }
            int offset = 0, col_type = 0;
            for (size_t i = 0; i < specified_channel_count;
                 i++, offset += col_type) {
                sstream >> col_type;
                header.fields[i].size = col_type;
                header.fields[i].offset = offset;
            }
            header.pointsize = offset;
        } else if (line_type.substr(0, 4) == "TYPE") {
            if (specified_channel_count != st.size() - 1) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.\n");
                return false;
            }
            for (size_t i = 0; i < specified_channel_count; i++) {
                header.fields[i].type = st[i + 1].c_str()[0];
            }
        } else if (line_type.substr(0, 5) == "COUNT") {
            if (specified_channel_count != st.size() - 1) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.\n");
                return false;
            }
            int count_offset = 0, offset = 0, col_count = 0;
            for (size_t i = 0; i < specified_channel_count; i++) {
                sstream >> col_count;
                header.fields[i].count = col_count;
                header.fields[i].count_offset = count_offset;
                header.fields[i].offset = offset;
                count_offset += col_count;
                offset += col_count * header.fields[i].size;
            }
            header.elementnum = count_offset;
            header.pointsize = offset;
        } else if (line_type.substr(0, 5) == "WIDTH") {
            sstream >> header.width;
        } else if (line_type.substr(0, 6) == "HEIGHT") {
            sstream >> header.height;
            header.points = header.width * header.height;
        } else if (line_type.substr(0, 9) == "VIEWPOINT") {
            if (st.size() >= 2) {
                header.viewpoint = st[1];
            }
        } else if (line_type.substr(0, 6) == "POINTS") {
            sstream >> header.points;
        } else if (line_type.substr(0, 4) == "DATA") {
            header.datatype = PCD_DATA_ASCII;
            if (st.size() >= 2) {
                if (st[1].substr(0, 17) == "binary_compressed") {
                    header.datatype = PCD_DATA_BINARY_COMPRESSED;
                } else if (st[1].substr(0, 6) == "binary") {
                    header.datatype = PCD_DATA_BINARY;
                }
            }
            break;
        }
    }
    if (CheckHeader(header) == false) {
        return false;
    }
    return true;
}

float UnpackBinaryPCDElement(const char *data_ptr,
                             const char type,
                             const int size) {
    if (type == 'I') {
        if (size == 1) {
            std::int8_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return (float)data;
        } else if (size == 2) {
            std::int16_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return (float)data;
        } else if (size == 4) {
            std::int32_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return (float)data;
        } else {
            return 0.0;
        }
    } else if (type == 'U') {
        if (size == 1) {
            std::uint8_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return (float)data;
        } else if (size == 2) {
            std::uint16_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return (float)data;
        } else if (size == 4) {
            std::uint32_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return (float)data;
        } else {
            return 0.0;
        }
    } else if (type == 'F') {
        if (size == 4) {
            std::float_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return (float)data;
        } else {
            return 0.0;
        }
    }
    return 0.0;
}

Eigen::Vector3f UnpackBinaryPCDColor(const char *data_ptr,
                                     const char type,
                                     const int size) {
    if (size == 4) {
        std::uint8_t data[4];
        memcpy(data, data_ptr, 4);
        // color data is packed in BGR order.
        return Eigen::Vector3f((float)data[2] / 255.0, (float)data[1] / 255.0,
                               (float)data[0] / 255.0);
    } else {
        return Eigen::Vector3f::Zero();
    }
}

float UnpackASCIIPCDElement(const char *data_ptr,
                            const char type,
                            const int size) {
    char *end;
    if (type == 'I') {
        return (float)std::strtol(data_ptr, &end, 0);
    } else if (type == 'U') {
        return (float)std::strtoul(data_ptr, &end, 0);
    } else if (type == 'F') {
        return std::strtod(data_ptr, &end);
    }
    return 0.0;
}

Eigen::Vector3f UnpackASCIIPCDColor(const char *data_ptr,
                                    const char type,
                                    const int size) {
    if (size == 4) {
        std::uint8_t data[4] = {0, 0, 0, 0};
        char *end;
        if (type == 'I') {
            std::int32_t value = std::strtol(data_ptr, &end, 0);
            memcpy(data, &value, 4);
        } else if (type == 'U') {
            std::uint32_t value = std::strtoul(data_ptr, &end, 0);
            memcpy(data, &value, 4);
        } else if (type == 'F') {
            std::float_t value = std::strtof(data_ptr, &end);
            memcpy(data, &value, 4);
        }
        return Eigen::Vector3f((float)data[2] / 255.0, (float)data[1] / 255.0,
                               (float)data[0] / 255.0);
    } else {
        return Eigen::Vector3f::Zero();
    }
}

bool ReadPCDData(FILE *file,
                 const PCDHeader &header,
                 geometry::PointCloud &pointcloud) {
    HostPointCloud host_pc;
    // The header should have been checked
    if (header.has_points) {
        host_pc.points_.resize(header.points);
    } else {
        utility::LogWarning(
                "[ReadPCDData] Fields for point data are not complete.\n");
        return false;
    }
    if (header.has_normals) {
        host_pc.normals_.resize(header.points);
    }
    if (header.has_colors) {
        host_pc.colors_.resize(header.points);
    }
    if (header.datatype == PCD_DATA_ASCII) {
        char line_buffer[DEFAULT_IO_BUFFER_SIZE];
        int idx = 0;
        while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file) &&
               idx < header.points) {
            std::string line(line_buffer);
            std::vector<std::string> strs;
            utility::SplitString(strs, line, "\t\r\n ");
            if ((int)strs.size() < header.elementnum) {
                continue;
            }
            for (size_t i = 0; i < header.fields.size(); i++) {
                const auto &field = header.fields[i];
                if (field.name == "x") {
                    host_pc.points_[idx](0) = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "y") {
                    host_pc.points_[idx](1) = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "z") {
                    host_pc.points_[idx](2) = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "normal_x") {
                    host_pc.normals_[idx](0) = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "normal_y") {
                    host_pc.normals_[idx](1) = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "normal_z") {
                    host_pc.normals_[idx](2) = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "rgb" || field.name == "rgba") {
                    host_pc.colors_[idx] = UnpackASCIIPCDColor(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                }
            }
            idx++;
        }
    } else if (header.datatype == PCD_DATA_BINARY) {
        std::unique_ptr<char[]> buffer(new char[header.pointsize]);
        for (int i = 0; i < header.points; i++) {
            if (fread(buffer.get(), header.pointsize, 1, file) != 1) {
                utility::LogWarning(
                        "[ReadPCDData] Failed to read data record.\n");
                pointcloud.Clear();
                return false;
            }
            for (const auto &field : header.fields) {
                if (field.name == "x") {
                    host_pc.points_[i](0) =
                            UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                   field.type, field.size);
                } else if (field.name == "y") {
                    host_pc.points_[i](1) =
                            UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                   field.type, field.size);
                } else if (field.name == "z") {
                    host_pc.points_[i](2) =
                            UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                   field.type, field.size);
                } else if (field.name == "normal_x") {
                    host_pc.normals_[i](0) =
                            UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                   field.type, field.size);
                } else if (field.name == "normal_y") {
                    host_pc.normals_[i](1) =
                            UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                   field.type, field.size);
                } else if (field.name == "normal_z") {
                    host_pc.normals_[i](2) =
                            UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                   field.type, field.size);
                } else if (field.name == "rgb" || field.name == "rgba") {
                    host_pc.colors_[i] =
                            UnpackBinaryPCDColor(buffer.get() + field.offset,
                                                 field.type, field.size);
                }
            }
        }
    } else if (header.datatype == PCD_DATA_BINARY_COMPRESSED) {
        std::uint32_t compressed_size;
        std::uint32_t uncompressed_size;
        if (fread(&compressed_size, sizeof(compressed_size), 1, file) != 1) {
            utility::LogWarning("[ReadPCDData] Failed to read data record.\n");
            pointcloud.Clear();
            return false;
        }
        if (fread(&uncompressed_size, sizeof(uncompressed_size), 1, file) !=
            1) {
            utility::LogWarning("[ReadPCDData] Failed to read data record.\n");
            pointcloud.Clear();
            return false;
        }
        utility::LogWarning(
                "PCD data with {:d} compressed size, and {:d} uncompressed "
                "size.\n",
                compressed_size, uncompressed_size);
        std::unique_ptr<char[]> buffer_compressed(new char[compressed_size]);
        if (fread(buffer_compressed.get(), 1, compressed_size, file) !=
            compressed_size) {
            utility::LogWarning("[ReadPCDData] Failed to read data record.\n");
            pointcloud.Clear();
            return false;
        }
        std::unique_ptr<char[]> buffer(new char[uncompressed_size]);
        if (lzf_decompress(buffer_compressed.get(),
                           (unsigned int)compressed_size, buffer.get(),
                           (unsigned int)uncompressed_size) !=
            uncompressed_size) {
            utility::LogWarning("[ReadPCDData] Uncompression failed.\n");
            pointcloud.Clear();
            return false;
        }
        for (const auto &field : header.fields) {
            const char *base_ptr = buffer.get() + field.offset * header.points;
            if (field.name == "x") {
                for (int i = 0; i < header.points; i++) {
                    host_pc.points_[i](0) = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "y") {
                for (int i = 0; i < header.points; i++) {
                    host_pc.points_[i](1) = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "z") {
                for (int i = 0; i < header.points; i++) {
                    host_pc.points_[i](2) = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "normal_x") {
                for (int i = 0; i < header.points; i++) {
                    host_pc.normals_[i](0) = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "normal_y") {
                for (int i = 0; i < header.points; i++) {
                    host_pc.normals_[i](1) = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "normal_z") {
                for (int i = 0; i < header.points; i++) {
                    host_pc.normals_[i](2) = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "rgb" || field.name == "rgba") {
                for (int i = 0; i < header.points; i++) {
                    host_pc.colors_[i] = UnpackBinaryPCDColor(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            }
        }
    }
    host_pc.ToDevice(pointcloud);
    return true;
}

bool GenerateHeader(const geometry::PointCloud &pointcloud,
                    const bool write_ascii,
                    const bool compressed,
                    PCDHeader &header) {
    if (pointcloud.HasPoints() == false) {
        return false;
    }
    header.version = "0.7";
    header.width = (int)pointcloud.points_.size();
    header.height = 1;
    header.points = header.width;
    header.fields.clear();
    PCLPointField field;
    field.type = 'F';
    field.size = 4;
    field.count = 1;
    field.name = "x";
    header.fields.push_back(field);
    field.name = "y";
    header.fields.push_back(field);
    field.name = "z";
    header.fields.push_back(field);
    header.elementnum = 3;
    header.pointsize = 12;
    if (pointcloud.HasNormals()) {
        field.name = "normal_x";
        header.fields.push_back(field);
        field.name = "normal_y";
        header.fields.push_back(field);
        field.name = "normal_z";
        header.fields.push_back(field);
        header.elementnum += 3;
        header.pointsize += 12;
    }
    if (pointcloud.HasColors()) {
        field.name = "rgb";
        header.fields.push_back(field);
        header.elementnum++;
        header.pointsize += 4;
    }
    if (write_ascii) {
        header.datatype = PCD_DATA_ASCII;
    } else {
        if (compressed) {
            header.datatype = PCD_DATA_BINARY_COMPRESSED;
        } else {
            header.datatype = PCD_DATA_BINARY;
        }
    }
    return true;
}

bool WritePCDHeader(FILE *file, const PCDHeader &header) {
    fprintf(file, "# .PCD v%s - Point Cloud Data file format\n",
            header.version.c_str());
    fprintf(file, "VERSION %s\n", header.version.c_str());
    fprintf(file, "FIELDS");
    for (const auto &field : header.fields) {
        fprintf(file, " %s", field.name.c_str());
    }
    fprintf(file, "\n");
    fprintf(file, "SIZE");
    for (const auto &field : header.fields) {
        fprintf(file, " %d", field.size);
    }
    fprintf(file, "\n");
    fprintf(file, "TYPE");
    for (const auto &field : header.fields) {
        fprintf(file, " %c", field.type);
    }
    fprintf(file, "\n");
    fprintf(file, "COUNT");
    for (const auto &field : header.fields) {
        fprintf(file, " %d", field.count);
    }
    fprintf(file, "\n");
    fprintf(file, "WIDTH %d\n", header.width);
    fprintf(file, "HEIGHT %d\n", header.height);
    fprintf(file, "VIEWPOINT 0 0 0 1 0 0 0\n");
    fprintf(file, "POINTS %d\n", header.points);

    switch (header.datatype) {
        case PCD_DATA_BINARY:
            fprintf(file, "DATA binary\n");
            break;
        case PCD_DATA_BINARY_COMPRESSED:
            fprintf(file, "DATA binary_compressed\n");
            break;
        case PCD_DATA_ASCII:
        default:
            fprintf(file, "DATA ascii\n");
            break;
    }
    return true;
}

float ConvertRGBToFloat(const Eigen::Vector3f &color) {
    std::uint8_t rgba[4] = {0, 0, 0, 0};
    rgba[2] = (std::uint8_t)std::max(std::min((int)(color(0) * 255.0), 255), 0);
    rgba[1] = (std::uint8_t)std::max(std::min((int)(color(1) * 255.0), 255), 0);
    rgba[0] = (std::uint8_t)std::max(std::min((int)(color(2) * 255.0), 255), 0);
    float value;
    memcpy(&value, rgba, 4);
    return value;
}

bool WritePCDData(FILE *file,
                  const PCDHeader &header,
                  const geometry::PointCloud &pointcloud) {
    bool has_normal = pointcloud.HasNormals();
    bool has_color = pointcloud.HasColors();
    HostPointCloud host_pc;
    host_pc.FromDevice(pointcloud);
    if (header.datatype == PCD_DATA_ASCII) {
        for (size_t i = 0; i < host_pc.points_.size(); i++) {
            const auto &point = host_pc.points_[i];
            fprintf(file, "%.10g %.10g %.10g", point(0), point(1), point(2));
            if (has_normal) {
                const auto &normal = host_pc.normals_[i];
                fprintf(file, " %.10g %.10g %.10g", normal(0), normal(1),
                        normal(2));
            }
            if (has_color) {
                const auto &color = host_pc.colors_[i];
                fprintf(file, " %.10g", ConvertRGBToFloat(color));
            }
            fprintf(file, "\n");
        }
    } else if (header.datatype == PCD_DATA_BINARY) {
        std::unique_ptr<float[]> data(new float[header.elementnum]);
        for (size_t i = 0; i < host_pc.points_.size(); i++) {
            const auto &point = host_pc.points_[i];
            data[0] = (float)point(0);
            data[1] = (float)point(1);
            data[2] = (float)point(2);
            int idx = 3;
            if (has_normal) {
                const auto &normal = host_pc.normals_[i];
                data[idx + 0] = (float)normal(0);
                data[idx + 1] = (float)normal(1);
                data[idx + 2] = (float)normal(2);
                idx += 3;
            }
            if (has_color) {
                const auto &color = host_pc.colors_[i];
                data[idx] = ConvertRGBToFloat(color);
            }
            fwrite(data.get(), sizeof(float), header.elementnum, file);
        }
    } else if (header.datatype == PCD_DATA_BINARY_COMPRESSED) {
        int strip_size = header.points;
        std::uint32_t buffer_size =
                (std::uint32_t)(header.elementnum * header.points);
        std::unique_ptr<float[]> buffer(new float[buffer_size]);
        std::unique_ptr<float[]> buffer_compressed(new float[buffer_size * 2]);
        for (size_t i = 0; i < host_pc.points_.size(); i++) {
            const auto &point = host_pc.points_[i];
            buffer[0 * strip_size + i] = (float)point(0);
            buffer[1 * strip_size + i] = (float)point(1);
            buffer[2 * strip_size + i] = (float)point(2);
            int idx = 3;
            if (has_normal) {
                const auto &normal = host_pc.normals_[i];
                buffer[(idx + 0) * strip_size + i] = (float)normal(0);
                buffer[(idx + 1) * strip_size + i] = (float)normal(1);
                buffer[(idx + 2) * strip_size + i] = (float)normal(2);
                idx += 3;
            }
            if (has_color) {
                const auto &color = host_pc.colors_[i];
                buffer[idx * strip_size + i] = ConvertRGBToFloat(color);
            }
        }
        std::uint32_t buffer_size_in_bytes = buffer_size * sizeof(float);
        std::uint32_t size_compressed =
                lzf_compress(buffer.get(), buffer_size_in_bytes,
                             buffer_compressed.get(), buffer_size_in_bytes * 2);
        if (size_compressed == 0) {
            utility::LogWarning("[WritePCDData] Failed to compress data.\n");
            return false;
        }
        utility::LogDebug(
                "[WritePCDData] {:d} bytes data compressed into {:d} bytes.\n",
                buffer_size_in_bytes, size_compressed);
        fwrite(&size_compressed, sizeof(size_compressed), 1, file);
        fwrite(&buffer_size_in_bytes, sizeof(buffer_size_in_bytes), 1, file);
        fwrite(buffer_compressed.get(), 1, size_compressed, file);
    }
    return true;
}

}  // unnamed namespace

namespace io {
bool ReadPointCloudFromPCD(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           bool print_progress) {
    PCDHeader header;
    FILE *file = fopen(filename.c_str(), "rb");
    if (file == NULL) {
        utility::LogWarning("Read PCD failed: unable to open file: {}\n",
                            filename);
        return false;
    }
    if (ReadPCDHeader(file, header) == false) {
        utility::LogWarning("Read PCD failed: unable to parse header.\n");
        fclose(file);
        return false;
    }
    utility::LogDebug(
            "PCD header indicates {:d} fields, {:d} bytes per point, and {:d} "
            "points "
            "in total.\n",
            (int)header.fields.size(), header.pointsize, header.points);
    for (const auto &field : header.fields) {
        utility::LogDebug("{}, {}, {:d}, {:d}, {:d}\n", field.name.c_str(),
                          field.type, field.size, field.count, field.offset);
    }
    utility::LogDebug("Compression method is {:d}.\n", (int)header.datatype);
    utility::LogDebug("Points: {};  normals: {};  colors: {}\n",
                      header.has_points ? "yes" : "no",
                      header.has_normals ? "yes" : "no",
                      header.has_colors ? "yes" : "no");
    if (ReadPCDData(file, header, pointcloud) == false) {
        utility::LogWarning("Read PCD failed: unable to read data.\n");
        fclose(file);
        return false;
    }
    fclose(file);
    return true;
}

bool WritePointCloudToPCD(const std::string &filename,
                          const geometry::PointCloud &pointcloud,
                          bool write_ascii /* = false*/,
                          bool compressed /* = false*/,
                          bool print_progress) {
    PCDHeader header;
    if (GenerateHeader(pointcloud, write_ascii, compressed, header) == false) {
        utility::LogWarning("Write PCD failed: unable to generate header.\n");
        return false;
    }
    FILE *file = fopen(filename.c_str(), "wb");
    if (file == NULL) {
        utility::LogWarning("Write PCD failed: unable to open file.\n");
        return false;
    }
    if (WritePCDHeader(file, header) == false) {
        utility::LogWarning("Write PCD failed: unable to write header.\n");
        fclose(file);
        return false;
    }
    if (WritePCDData(file, header, pointcloud) == false) {
        utility::LogWarning("Write PCD failed: unable to write data.\n");
        fclose(file);
        return false;
    }
    fclose(file);
    return true;
}

}  // namespace io
}  // namespace cupoch


#include <omp.h>
#include <future>
#include <functional>

#define BILLION  1000000000.0
int main(int argc, char *argv[]){
    auto starttt = std::chrono::system_clock::now();


	using namespace cupoch;                                                                                                                                                                                         
  using namespace cupoch::registration;
	utility::InitializeAllocator();

	//utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);                                                                                                                                                     
	if (argc < 4) {utility::LogInfo("Need two arguments of point cloud file name."); return 0;}


	int total_iteration = 28;
	FILE *pose_data = fopen(argv[3], "rb"); 
	std::vector<geometry::PointCloud> est_cloud_arr_test(total_iteration);                                                    

	//geometry::PointCloud est_cloud_arr[total_iteration];
	//for (int i=0; i<total_iteration; i++){
	//	est_cloud_arr[i] = geometry::PointCloud();
	//}
	
	geometry::PointCloud est_cloud = geometry::PointCloud();
	geometry::PointCloud obj_cloud = geometry::PointCloud();
	//Eigen::Matrix4f init[total_iteration];
	std::vector<Eigen::Matrix4f> init_test(total_iteration);
	double val;


	PCDHeader header;

	FILE *fileIn = fopen(argv[2], "rb"); /* Reading obj_cloud */ 
	if (fileIn == NULL){
		std::cout << "Error in oprning file : " << argv[1] << "\n";
	}

	if (ReadPCDHeader(fileIn, header) == false) {
		std::cout << "Failed to read header\n";
		fclose(fileIn);
		return -1;
	}
	if (ReadPCDData(fileIn, header, obj_cloud) == false){
		std::cout << "Error in reading data\n";
		fclose(fileIn);
		return -1;
	}

	fclose(fileIn);

	fileIn = fopen(argv[1], "rb");  /* Reading estimate_cloud */ 
	if (fileIn == NULL){
		std::cout << "Error in oprning file : " << argv[1] << "\n";
	}

	int total_size = 0;
	for(int iteration=0; iteration < total_iteration; iteration++){

		if (ReadPCDHeader(fileIn, header) == false) {
			std::cout << "Failed to read header\n";
			fclose(fileIn);
			return -1;
		}
		//if (ReadPCDData(fileIn, header, est_cloud_arr[iteration]) == false){
		//	std::cout << "Error in reading data\n";
		//	fclose(fileIn);
		//	return -1;
		//}
		if (ReadPCDData(fileIn, header, est_cloud_arr_test[iteration]) == false){
                        std::cout << "Error in reading data\n";
                        fclose(fileIn);
                        return -1;
                }

		total_size += est_cloud_arr_test[iteration].points_.size();

		for (int i=0; i<4; i++){
			for(int j=0; j<4; j++){
				fread(&val, sizeof(double), 1, pose_data);
				//init[iteration](i,j) = float(val);
				init_test[iteration](i,j) = float(val);		
			}
		}
	}

	std::cout << "Total points : " << total_size << "\n";


	struct timespec start, end;
	double time_spent;
	bool res;

#if 1	
	{

		clock_gettime(CLOCK_REALTIME, &start);
		
		geometry::KDTreeFlann kdtree(obj_cloud);
		auto res = registration::AkashaRegistrationICP(est_cloud_arr_test, obj_cloud, 1.0, kdtree, total_iteration, init_test, TransformationEstimationPointToPoint(),ICPConvergenceCriteria(1e-6,1e-6,100));
		for(int iteration = 0; iteration < total_iteration; iteration+=1){
                std::cout << res[iteration].transformation_ << std::endl;
        	}
	
	
		clock_gettime(CLOCK_REALTIME, &end);
		time_spent = (end.tv_sec - start.tv_sec) +
			(end.tv_nsec - start.tv_nsec) / BILLION;
		printf("Time elpased is with Optimization is : %f seconds\n", time_spent);
	}

#else	
	clock_gettime(CLOCK_REALTIME, &start);
	for(int iteration=0; iteration < total_iteration; iteration++){
		auto res = registration::RegistrationICP(est_cloud_arr[iteration], obj_cloud, 1.0, init[iteration], TransformationEstimationPointToPoint(),ICPConvergenceCriteria(1e-6,1e-6,100));
    std::cout << res.transformation_ << std::endl;
	}
	
	clock_gettime(CLOCK_REALTIME, &end);
	time_spent = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;
	printf("Time elpased with base case is : %f seconds\n", time_spent);

#endif	

	fclose(fileIn);
    auto endtt = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = endtt-starttt;
    std::time_t end_time = std::chrono::system_clock::to_time_t(endtt);
    std::cout << "Time taken for total app" << elapsed_seconds.count() << " seconds.\n";
	return 0;
}

