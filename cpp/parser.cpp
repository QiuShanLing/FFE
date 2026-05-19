#include <iostream>
#include <fstream> 
#include <string>
#include <string_view>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <set>
#include <charconv> // C++17用于快速转换数字
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// ==========================================
// 1. 数据结构
// ==========================================
struct Section {
    double frequency;           // 当前 Section 的频率
    std::vector<double> data;   // 扁平化的数据 (Rows * Cols)
    size_t row_count;           // 行数
};

struct FFEFile {
    std::vector<std::string> headers; // 列名 (如 Theta, Phi, Re(E)...)
    std::vector<Section> sections;    // 所有 Section 数据
    size_t axis1_samples = 0;
    size_t axis2_samples = 0;
};

// ==========================================
// 2. 辅助工具
// ==========================================
std::string read_file_to_string(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("无法打开文件: " + path);
    auto size = file.tellg();  // 获取文件大小(当前指针在文件末尾)
    std::string content(size, '\0');  // 创建空字符串
    file.seekg(0);  // 将读取指针调到文件头
    if (file.read(&content[0], size)) return content;
    return {};
}

// 快速解析一行 Header (去除 # 和 空格)
std::vector<std::string> parse_header_line(const char* start, const char* end) {
    std::vector<std::string> headers;
    std::string current;
    for (const char* p = start; p < end; ++p) {
        char c = *p;
        // 跳过注释符、引号、换行
        if (c == '#' || c == '"' || c == '\r' || c == '\n') continue;
        
        // 遇到分隔符 (空格, Tab, 逗号)
        if (std::isspace(static_cast<unsigned char>(c)) || c == ',') {
            if (!current.empty()) {
                headers.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) headers.push_back(current);
    return headers;
}

void parse_sample_counts(const std::string& content, FFEFile& result) {
    const char* ptr = content.data();
    const char* end = content.data() + content.size();
    const char* key = "#No. of ";
    const size_t key_len = std::strlen(key);
    const char* suffix = " Samples:";
    const size_t suffix_len = std::strlen(suffix);

    while (ptr < end) {
        auto line_end = std::find(ptr, end, '\n');
        auto key_pos = std::search(ptr, line_end, key, key + key_len);
        if (key_pos != line_end) {
            const char* name_start = key_pos + key_len;
            auto suffix_pos = std::search(name_start, line_end, suffix, suffix + suffix_len);
            if (suffix_pos != line_end) {
                std::string axis_name(name_start, suffix_pos);
                size_t value = static_cast<size_t>(std::strtoull(suffix_pos + suffix_len, nullptr, 10));

                if (!result.headers.empty()) {
                    if (axis_name == result.headers[0]) {
                        result.axis1_samples = value;
                    } else if (result.headers.size() > 1 && axis_name == result.headers[1]) {
                        result.axis2_samples = value;
                    }
                } else if (axis_name == "Theta" || axis_name == "U") {
                    result.axis1_samples = value;
                } else if (axis_name == "Phi" || axis_name == "V") {
                    result.axis2_samples = value;
                }
            }
        }
        ptr = line_end == end ? end : line_end + 1;
    }
}

// ==========================================
// 3. 核心解析函数 (包含 Header 和 Frequency)
// ==========================================
FFEFile parse_content(const std::string& content) {
    FFEFile result;
    const char* ptr = content.data();  // 初始指针
    const char* end = content.data() + content.size();

    // --- A. 解析 Header ---
    // 逻辑：找到包含 "Theta" 的那一行
    const char* HEADER_KEY = "\"Theta\"";
    auto header_pos = std::search(ptr, end, HEADER_KEY, HEADER_KEY + 7);  // 返回Header行的起始位置指针
    
    if (header_pos != end) {
        // 向前找行首
        const char* line_start = header_pos;
        while (line_start > ptr && *line_start != '\n') line_start--;
        if (*line_start == '\n') line_start++;

        // 向后找行尾
        const char* line_end = header_pos;
        while (line_end < end && *line_end != '\n') line_end++;

        result.headers = parse_header_line(line_start, line_end);
        
        // 解析指针移动到 Header 之后，避免重复解析
        // 已修复，调到Header之后会跳过第一个Section，直接查找第二个
        // ptr = line_end; 
    }

    parse_sample_counts(content, result);

    // --- B. 解析 Sections ---
    // 根据您的 Python 代码，Section 由 #Configuration Name 分隔
    const char* SECTION_TAG = "#Configuration Name";
    const size_t TAG_LEN = std::strlen(SECTION_TAG);
    const char* FREQ_TAG = "#Frequency:";
    const size_t FREQ_TAG_LEN = std::strlen(FREQ_TAG);

    while (ptr < end) {
        // 1. 查找下一个 Section
        auto section_start = std::search(ptr, end, SECTION_TAG, SECTION_TAG + TAG_LEN);
        if (section_start == end) break;

        // 移动指针，开始处理这个 Section
        // ptr = section_start + TAG_LEN;

        // 找到 Section 后，直接跳到这一行的末尾换行符处
        ptr = section_start;
        while (ptr < end && *ptr != '\n') ptr++;
        
        // 确定这个 Section 的结束位置（下一个 Section 的开始，或者文件末尾）
        auto next_section = std::search(ptr, end, SECTION_TAG, SECTION_TAG + TAG_LEN);
        
        // 这一段数据的范围
        const char* current_ptr = ptr;
        const char* section_end = next_section;

        Section current_sec;
        current_sec.frequency = 0.0;
        current_sec.data.reserve(2000); // 预分配内存，假设大概有几千个数

        // --- C. 在当前 Section 内查找频率 ---
        auto freq_pos = std::search(current_ptr, section_end, FREQ_TAG, FREQ_TAG + FREQ_TAG_LEN);
        if (freq_pos != section_end) {
            // 找到频率，读取后面的数字
            char* val_end;
            current_sec.frequency = std::strtod(freq_pos + FREQ_TAG_LEN, &val_end);
        }

        // --- D. 解析数值 ---
        while (current_ptr < section_end) {
            // 跳过空白字符
            while (current_ptr < section_end && std::isspace(static_cast<unsigned char>(*current_ptr))) {
                current_ptr++;
            }
            
            // 简单过滤：如果这一行是以 '#' 开头（除了 #Frequency 上面已经处理了），整行跳过
            // 注意：这会防止把注释里的数字读进去
            if (*current_ptr == '#') {
                while (current_ptr < section_end && *current_ptr != '\n') current_ptr++;
                continue;
            }

            if (current_ptr >= section_end) break;

            // 尝试解析数字
            char c = *current_ptr;
            // 判断是否是数字字符 (包括负号和小数点)
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.') {
                char* num_end;
                double val = std::strtod(current_ptr, &num_end);
                
                if (current_ptr == num_end) {
                    current_ptr++; // 解析失败，强制步进
                } else {
                    current_sec.data.push_back(val);
                    current_ptr = num_end;
                }
            } else {
                current_ptr++;
            }
        }

        // 计算行数 (总数据量 / 列数)
        if (!result.headers.empty()) {
            current_sec.row_count = current_sec.data.size() / result.headers.size();
        }

        result.sections.push_back(std::move(current_sec));
        
        // 准备下一轮循环
        ptr = next_section;
    }

    return result;
}

FFEFile parse_ffe(const std::string path){
    std::string content = read_file_to_string(path);
    FFEFile file_data = parse_content(content);
    return file_data;
}

py::array_t<double> section_data_array(const Section& section) {
    if (section.data.empty() || section.row_count == 0) {
        return py::array_t<double>();
    }

    size_t rows = section.row_count;
    size_t cols = section.data.size() / rows;
    std::vector<py::ssize_t> shape(2);
    shape[0] = static_cast<py::ssize_t>(rows);
    shape[1] = static_cast<py::ssize_t>(cols);

    py::array_t<double> arr(shape);
    std::memcpy(arr.mutable_data(), section.data.data(), section.data.size() * sizeof(double));
    return arr;
}

py::tuple parse_ffe_array(const std::string path) {
    FFEFile file_data = parse_ffe(path);

    if (file_data.headers.empty()) {
        throw std::runtime_error("FFE header was not found.");
    }
    if (file_data.sections.empty()) {
        throw std::runtime_error("No FFE sections were found.");
    }

    const size_t n_freq = file_data.sections.size();
    const size_t n_cols = file_data.headers.size();
    const size_t n_rows = file_data.sections[0].row_count;

    if (n_rows == 0) {
        throw std::runtime_error("First FFE section contains no data rows.");
    }

    std::vector<py::ssize_t> freq_shape(1);
    freq_shape[0] = static_cast<py::ssize_t>(n_freq);
    py::array_t<double> freqs(freq_shape);

    std::vector<py::ssize_t> data_shape(3);
    data_shape[0] = static_cast<py::ssize_t>(n_freq);
    data_shape[1] = static_cast<py::ssize_t>(n_rows);
    data_shape[2] = static_cast<py::ssize_t>(n_cols);
    py::array_t<double> data(data_shape);

    auto freqs_view = freqs.mutable_unchecked<1>();
    auto data_view = data.mutable_unchecked<3>();

    for (size_t i = 0; i < n_freq; ++i) {
        const Section& section = file_data.sections[i];
        if (section.row_count != n_rows) {
            throw std::runtime_error("FFE sections have inconsistent row counts.");
        }
        if (section.data.size() != n_rows * n_cols) {
            throw std::runtime_error("FFE section data size does not match header column count.");
        }

        freqs_view(i) = section.frequency;
        const double* src = section.data.data();
        for (size_t r = 0; r < n_rows; ++r) {
            for (size_t c = 0; c < n_cols; ++c) {
                data_view(i, r, c) = src[r * n_cols + c];
            }
        }
    }

    return py::make_tuple(file_data.headers, freqs, data);
}

std::vector<double> infer_axis_values(const Section& section, size_t col, size_t n_cols) {
    std::set<double> values;
    for (size_t row = 0; row < section.row_count; ++row) {
        values.insert(section.data[row * n_cols + col]);
    }
    return std::vector<double>(values.begin(), values.end());
}

bool validate_axis2_inner(
    const Section& section,
    size_t n_cols,
    size_t n_axis1,
    size_t n_axis2,
    const std::vector<double>& axis1,
    const std::vector<double>& axis2
) {
    for (size_t i = 0; i < n_axis1; ++i) {
        for (size_t j = 0; j < n_axis2; ++j) {
            const size_t row = i * n_axis2 + j;
            if (
                section.data[row * n_cols] != axis1[i] ||
                section.data[row * n_cols + 1] != axis2[j]
            ) {
                return false;
            }
        }
    }
    return true;
}

bool validate_axis1_inner(
    const Section& section,
    size_t n_cols,
    size_t n_axis1,
    size_t n_axis2,
    const std::vector<double>& axis1,
    const std::vector<double>& axis2
) {
    for (size_t j = 0; j < n_axis2; ++j) {
        for (size_t i = 0; i < n_axis1; ++i) {
            const size_t row = j * n_axis1 + i;
            if (
                section.data[row * n_cols] != axis1[i] ||
                section.data[row * n_cols + 1] != axis2[j]
            ) {
                return false;
            }
        }
    }
    return true;
}

py::tuple parse_ffe_grid(const std::string path) {
    FFEFile file_data = parse_ffe(path);

    if (file_data.headers.size() < 2) {
        throw std::runtime_error("FFE header must contain at least two coordinate columns.");
    }
    if (file_data.sections.empty()) {
        throw std::runtime_error("No FFE sections were found.");
    }

    const size_t n_freq = file_data.sections.size();
    const size_t n_cols = file_data.headers.size();
    const size_t n_rows = file_data.sections[0].row_count;

    if (n_rows == 0) {
        throw std::runtime_error("First FFE section contains no data rows.");
    }

    std::vector<double> axis1 = infer_axis_values(file_data.sections[0], 0, n_cols);
    std::vector<double> axis2 = infer_axis_values(file_data.sections[0], 1, n_cols);
    size_t n_axis1 = file_data.axis1_samples == 0 ? axis1.size() : file_data.axis1_samples;
    size_t n_axis2 = file_data.axis2_samples == 0 ? axis2.size() : file_data.axis2_samples;

    if (n_axis1 * n_axis2 != n_rows) {
        throw std::runtime_error("FFE axis sample counts do not match data row count.");
    }
    if (axis1.size() != n_axis1 || axis2.size() != n_axis2) {
        throw std::runtime_error("FFE coordinate values do not match axis sample counts.");
    }

    const bool axis2_inner = validate_axis2_inner(
        file_data.sections[0], n_cols, n_axis1, n_axis2, axis1, axis2
    );
    const bool axis1_inner = !axis2_inner && validate_axis1_inner(
        file_data.sections[0], n_cols, n_axis1, n_axis2, axis1, axis2
    );

    if (!axis2_inner && !axis1_inner) {
        throw std::runtime_error("FFE coordinate ordering is not a regular grid.");
    }

    std::vector<py::ssize_t> freq_shape(1);
    freq_shape[0] = static_cast<py::ssize_t>(n_freq);
    py::array_t<double> freqs(freq_shape);

    std::vector<py::ssize_t> axis1_shape(1);
    axis1_shape[0] = static_cast<py::ssize_t>(n_axis1);
    py::array_t<double> axis1_array(axis1_shape);

    std::vector<py::ssize_t> axis2_shape(1);
    axis2_shape[0] = static_cast<py::ssize_t>(n_axis2);
    py::array_t<double> axis2_array(axis2_shape);

    std::vector<py::ssize_t> data_shape(4);
    data_shape[0] = static_cast<py::ssize_t>(n_freq);
    data_shape[1] = static_cast<py::ssize_t>(n_axis1);
    data_shape[2] = static_cast<py::ssize_t>(n_axis2);
    data_shape[3] = static_cast<py::ssize_t>(n_cols);
    py::array_t<double> data(data_shape);

    std::memcpy(axis1_array.mutable_data(), axis1.data(), axis1.size() * sizeof(double));
    std::memcpy(axis2_array.mutable_data(), axis2.data(), axis2.size() * sizeof(double));

    auto freqs_view = freqs.mutable_unchecked<1>();
    auto data_view = data.mutable_unchecked<4>();

    for (size_t f = 0; f < n_freq; ++f) {
        const Section& section = file_data.sections[f];
        if (section.row_count != n_rows || section.data.size() != n_rows * n_cols) {
            throw std::runtime_error("FFE sections have inconsistent data shapes.");
        }

        freqs_view(f) = section.frequency;
        for (size_t i = 0; i < n_axis1; ++i) {
            for (size_t j = 0; j < n_axis2; ++j) {
                const size_t row = axis2_inner ? i * n_axis2 + j : j * n_axis1 + i;
                for (size_t c = 0; c < n_cols; ++c) {
                    data_view(f, i, j, c) = section.data[row * n_cols + c];
                }
            }
        }
    }

    return py::make_tuple(file_data.headers, freqs, axis1_array, axis2_array, data);
}


// 在 PYBIND11_MODULE 内部

PYBIND11_MODULE(_parser, m) {
    m.doc() = "Fast FFE parser with 2D Numpy support";

    py::class_<Section>(m, "Section")
        .def(py::init<>())
        .def_readwrite("frequency", &Section::frequency)
        .def_readwrite("row_count", &Section::row_count)
        .def_property_readonly("data", &section_data_array);

    py::class_<FFEFile>(m, "FFEFile")
        .def(py::init<>())
        .def_readwrite("headers", &FFEFile::headers)   
        .def_readwrite("sections", &FFEFile::sections);

    m.def("parse_ffe", &parse_ffe, "Parse FFE file");
    m.def("parse_ffe_array", &parse_ffe_array, "Parse FFE file and return headers, frequencies, and a 3D NumPy array");
    m.def("parse_ffe_grid", &parse_ffe_grid, "Parse FFE file and return grid-shaped arrays for xarray construction");
}
