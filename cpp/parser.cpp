#include <iostream>
#include <fstream> 
#include <string>
#include <string_view>
#include <chrono>
#include <cstring>
#include <algorithm>
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


// 在 PYBIND11_MODULE 内部

PYBIND11_MODULE(_parser, m) {
    m.doc() = "Fast FFE parser with 2D Numpy support";

    py::class_<Section>(m, "Section")
        .def(py::init<>())
        .def_readwrite("frequency", &Section::frequency)
        .def_readwrite("row_count", &Section::row_count)
        
        // 【核心修改】将一维 vector 直接映射为二维 Numpy 数组
        .def_property_readonly("data", [](const Section &s) {
            if (s.data.empty() || s.row_count == 0) {
                return py::array_t<double>(); // 空数组
            }

            // 1. 计算维度
            size_t rows = s.row_count;
            size_t cols = s.data.size() / rows; // 自动推算列数

            // 2. 定义 Numpy 的 Shape (行, 列)
            std::vector<size_t> shape = {rows, cols};

            // 3. 定义 Strides (步幅)
            // 意思是：要在内存中跳到下一行，需要跳过 cols * 8 字节
            //        要在内存中跳到下一列，需要跳过 8 字节 (double大小)
            std::vector<size_t> strides = {cols * sizeof(double), sizeof(double)};

            // 4. 返回数组（这是基于内存拷贝的，很安全）
            // 如果追求极致零拷贝，需要稍微复杂的 memory_view 处理，但通常拷贝一次足够快了
            return py::array_t<double>(shape, strides, s.data.data());
        });

    py::class_<FFEFile>(m, "FFEFile")
        .def(py::init<>())
        .def_readwrite("headers", &FFEFile::headers)   
        .def_readwrite("sections", &FFEFile::sections);

    m.def("parse_ffe", &parse_ffe, "Parse FFE file");
}
