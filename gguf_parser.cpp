#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <map>
#include <unordered_map>
#include <variant>
#include <cmath>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <iostream>

namespace gguf
{

    // Constants
    constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" in ASCII
    constexpr uint32_t GGUF_VERSION = 3;
    constexpr uint32_t GGUF_DEFAULT_ALIGNMENT = 32;
    const std::vector<uint32_t> READER_SUPPORTED_VERSIONS = {2, GGUF_VERSION};

    // GGML Quantization Types
    enum class GGMLQuantizationType : uint32_t
    {
        F32 = 0,
        F16 = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2_K = 10,
        Q3_K = 11,
        Q4_K = 12,
        Q5_K = 13,
        Q6_K = 14,
        Q8_K = 15,
        I8 = 24,
        I16 = 25,
        I32 = 26,
        I64 = 27,
        F64 = 28
    };

    // Metadata Value Types
    enum class GGUFValueType : uint32_t
    {
        UINT8 = 0,
        INT8 = 1,
        UINT16 = 2,
        INT16 = 3,
        UINT32 = 4,
        INT32 = 5,
        FLOAT32 = 6,
        BOOL = 7,
        STRING = 8,
        ARRAY = 9,
        UINT64 = 10,
        INT64 = 11,
        FLOAT64 = 12
    };

    // Quantization size information
    struct QuantSize
    {
        uint32_t block_size;
        uint32_t type_size;
    };

    const std::unordered_map<GGMLQuantizationType, QuantSize> GGML_QUANT_SIZES = {
        {GGMLQuantizationType::F32, {1, 4}},
        {GGMLQuantizationType::F16, {1, 2}},
        {GGMLQuantizationType::Q4_0, {32, 16}},
        {GGMLQuantizationType::Q4_1, {32, 18}},
        {GGMLQuantizationType::Q5_0, {32, 20}},
        {GGMLQuantizationType::Q5_1, {32, 20}},
        {GGMLQuantizationType::Q8_0, {32, 32}},
        {GGMLQuantizationType::Q8_1, {32, 32}},
        {GGMLQuantizationType::Q2_K, {256, 128}},
        {GGMLQuantizationType::Q3_K, {256, 192}},
        {GGMLQuantizationType::Q4_K, {256, 256}},
        {GGMLQuantizationType::Q5_K, {256, 320}},
        {GGMLQuantizationType::Q6_K, {256, 384}},
        {GGMLQuantizationType::Q8_K, {256, 512}},
        {GGMLQuantizationType::I8, {1, 1}},
        {GGMLQuantizationType::I16, {1, 2}},
        {GGMLQuantizationType::I32, {1, 4}},
        {GGMLQuantizationType::I64, {1, 8}},
        {GGMLQuantizationType::F64, {1, 8}}};

    // Forward declarations
    class ReaderField;
    class ReaderTensor;
    class ThreadPool;

    class ThreadPool
    {
    public:
        explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency())
            : stop(false)
        {
            for (size_t i = 0; i < num_threads; ++i)
            {
                workers.emplace_back([this]
                                     {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        if(stop && tasks.empty()) {
                            return;
                        }
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                } });
            }
        }

        template <class F>
        auto enqueue(F &&f) -> std::future<decltype(f())>
        {
            using return_type = decltype(f());
            auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
            std::future<return_type> res = task->get_future();
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (stop)
                {
                    throw std::runtime_error("enqueue on stopped ThreadPool");
                }
                tasks.emplace([task]()
                              { (*task)(); });
            }
            condition.notify_one();
            return res;
        }

        ~ThreadPool()
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = true;
            }
            condition.notify_all();
            for (std::thread &worker : workers)
            {
                worker.join();
            }
        }

    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex queue_mutex;
        std::condition_variable condition;
        bool stop;
    };

    using MetadataValue = std::variant<
        uint8_t, int8_t, uint16_t, int16_t,
        uint32_t, int32_t, float,
        bool, std::string,
        std::vector<uint8_t>, // For array data
        uint64_t, int64_t, double>;

    // Field information class
    class ReaderField
    {
    public:
        uint64_t offset;
        std::string name;
        std::vector<GGUFValueType> types;
        std::vector<MetadataValue> values;
        std::vector<int> data_indices;

        ReaderField() = default;
        ReaderField(uint64_t offs, std::string n)
            : offset(offs), name(std::move(n)) {}
    };

    // Tensor information class
    class ReaderTensor
    {
    public:
        std::string name;
        GGMLQuantizationType tensor_type;
        std::vector<uint64_t> shape;
        uint64_t n_elements;
        uint64_t n_bytes;
        uint64_t data_offset;
        std::vector<uint8_t> data;
        ReaderField field;

        ReaderTensor() = default;
        ReaderTensor(
            std::string n,
            GGMLQuantizationType type,
            std::vector<uint64_t> s,
            uint64_t elements,
            uint64_t bytes,
            uint64_t offset,
            std::vector<uint8_t> d,
            ReaderField f) : name(std::move(n)),
                             tensor_type(type),
                             shape(std::move(s)),
                             n_elements(elements),
                             n_bytes(bytes),
                             data_offset(offset),
                             data(std::move(d)),
                             field(std::move(f)) {}
    };

    // Main GGUF Reader class
    class GGUFReader
    {
        std::unique_ptr<ThreadPool> thread_pool;
        std::mutex file_mutex; // Para proteger el acceso al archivo

    public:
        explicit GGUFReader(const std::string &path, size_t num_threads = std::thread::hardware_concurrency())
        {
            file.open(path, std::ios::binary | std::ios::in);
            if (!file)
            {
                throw std::runtime_error("Cannot open file: " + path);
            }

            thread_pool = std::make_unique<ThreadPool>(num_threads);
            initialize();
        }

        ~GGUFReader()
        {
            if (file.is_open())
            {
                file.close();
            }
        }

        // Get a field by key
        const ReaderField *get_field(const std::string &key) const
        {
            auto it = fields.find(key);
            return it != fields.end() ? &it->second : nullptr;
        }

        // Get a tensor by index
        const ReaderTensor &get_tensor(size_t idx) const
        {
            if (idx >= tensors.size())
            {
                throw std::out_of_range("Tensor index out of range");
            }
            return tensors[idx];
        }

        // Get the number of tensors
        size_t get_tensor_count() const
        {
            return tensors.size();
        }

        // Get all field names
        std::vector<std::string> get_field_names() const
        {
            std::vector<std::string> names;
            names.reserve(fields.size());
            for (const auto &[name, _] : fields)
            {
                names.push_back(name);
            }
            return names;
        }

    private:
        std::ifstream file;
        bool is_little_endian;
        uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
        uint64_t data_offset = 0;

        std::map<std::string, ReaderField> fields;
        std::vector<ReaderTensor> tensors;

        // Helper to detect system endianness
        bool is_system_little_endian()
        {
            const uint16_t number = 0x1234;
            const uint8_t *ptr = reinterpret_cast<const uint8_t *>(&number);
            return ptr[0] == 0x34;
        }

        void initialize()
        {
            is_little_endian = is_system_little_endian();

            // Read and verify magic number
            uint32_t magic = read_value<uint32_t>();
            if (magic != GGUF_MAGIC)
            {
                throw std::runtime_error("Invalid GGUF magic number");
            }

            // Read and verify version
            uint32_t version = read_value<uint32_t>();
            if (std::find(READER_SUPPORTED_VERSIONS.begin(),
                          READER_SUPPORTED_VERSIONS.end(),
                          version) == READER_SUPPORTED_VERSIONS.end())
            {
                throw std::runtime_error("Unsupported GGUF version: " + std::to_string(version));
            }

            // Read counts
            uint64_t tensor_count = read_value<uint64_t>();
            uint64_t kv_count = read_value<uint64_t>();

            // Read metadata key-value pairs
            read_metadata(kv_count);

            // Update alignment if specified
            auto alignment_field = get_field("general.alignment");
            if (alignment_field)
            {
                if (alignment_field->types[0] == GGUFValueType::UINT32)
                {
                    alignment = std::get<uint32_t>(alignment_field->values[0]);
                }
            }

            // Read tensor information
            read_tensors(tensor_count);
        }

        template <typename T>
        T read_value()
        {
            T value;
            file.read(reinterpret_cast<char *>(&value), sizeof(T));
            return value;
        }

        std::string read_string()
        {
            uint64_t length = read_value<uint64_t>();
            std::string str(length, '\0');
            file.read(&str[0], length);
            return str;
        }

        void read_metadata(uint64_t count)
        {
            for (uint64_t i = 0; i < count; i++)
            {
                uint64_t offset = file.tellg();

                // Read key
                std::string key = read_string();

                // Read value type
                GGUFValueType value_type = static_cast<GGUFValueType>(read_value<uint32_t>());

                // Create field
                ReaderField field(offset, key);
                field.types.push_back(value_type);

                // Read value based on type
                read_field_value(field, value_type);

                // Store field
                fields[key] = std::move(field);
            }
        }

        void read_field_value(ReaderField &field, GGUFValueType type)
        {
            switch (type)
            {
            case GGUFValueType::UINT8:
                field.values.push_back(read_value<uint8_t>());
                break;
            case GGUFValueType::INT8:
                field.values.push_back(read_value<int8_t>());
                break;
            case GGUFValueType::UINT16:
                field.values.push_back(read_value<uint16_t>());
                break;
            case GGUFValueType::INT16:
                field.values.push_back(read_value<int16_t>());
                break;
            case GGUFValueType::UINT32:
                field.values.push_back(read_value<uint32_t>());
                break;
            case GGUFValueType::INT32:
                field.values.push_back(read_value<int32_t>());
                break;
            case GGUFValueType::FLOAT32:
                field.values.push_back(read_value<float>());
                break;
            case GGUFValueType::BOOL:
                field.values.push_back(static_cast<bool>(read_value<uint8_t>()));
                break;
            case GGUFValueType::STRING:
                field.values.push_back(read_string());
                break;
            case GGUFValueType::UINT64:
                field.values.push_back(read_value<uint64_t>());
                break;
            case GGUFValueType::INT64:
                field.values.push_back(read_value<int64_t>());
                break;
            case GGUFValueType::FLOAT64:
                field.values.push_back(read_value<double>());
                break;
            case GGUFValueType::ARRAY:
                read_array_value(field);
                break;
            default:
                throw std::runtime_error("Unknown value type: " + std::to_string(static_cast<int>(type)));
            }
        }

        void read_array_value(ReaderField &field)
        {
            // Read array type and length
            GGUFValueType element_type = static_cast<GGUFValueType>(read_value<uint32_t>());
            uint64_t length = read_value<uint64_t>();

            // Store the array type
            field.types.push_back(element_type);

            // Create a vector to hold array data
            std::vector<MetadataValue> array_values;
            array_values.reserve(length);

            // Read array elements
            for (uint64_t i = 0; i < length; i++)
            {
                switch (element_type)
                {
                case GGUFValueType::UINT8:
                    array_values.push_back(read_value<uint8_t>());
                    break;
                case GGUFValueType::INT8:
                    array_values.push_back(read_value<int8_t>());
                    break;
                case GGUFValueType::UINT16:
                    array_values.push_back(read_value<uint16_t>());
                    break;
                case GGUFValueType::INT16:
                    array_values.push_back(read_value<int16_t>());
                    break;
                case GGUFValueType::UINT32:
                    array_values.push_back(read_value<uint32_t>());
                    break;
                case GGUFValueType::INT32:
                    array_values.push_back(read_value<int32_t>());
                    break;
                case GGUFValueType::FLOAT32:
                    array_values.push_back(read_value<float>());
                    break;
                case GGUFValueType::BOOL:
                {
                    uint8_t val = read_value<uint8_t>();
                    array_values.push_back(static_cast<bool>(val));
                    break;
                }
                case GGUFValueType::STRING:
                    array_values.push_back(read_string());
                    break;
                case GGUFValueType::UINT64:
                    array_values.push_back(read_value<uint64_t>());
                    break;
                case GGUFValueType::INT64:
                    array_values.push_back(read_value<int64_t>());
                    break;
                case GGUFValueType::FLOAT64:
                    array_values.push_back(read_value<double>());
                    break;
                default:
                    throw std::runtime_error("Unsupported array element type: " +
                                             std::to_string(static_cast<int>(element_type)));
                }
            }

            field.values = std::move(array_values);
        }

        void read_tensors(uint64_t count)
        {
            std::vector<ReaderField> tensor_fields;
            // Read tensor information first
            for (uint64_t i = 0; i < count; i++)
            {
                uint64_t offset = file.tellg();
                ReaderField field = read_tensor_info(offset);
                tensor_fields.push_back(std::move(field));
            }

            // Align to the specified alignment
            uint64_t current_pos = file.tellg();
            data_offset = ((current_pos + alignment - 1) / alignment) * alignment;
            file.seekg(data_offset);

            // Create futures for parallel tensor reading
            std::vector<std::future<ReaderTensor>> futures;
            futures.reserve(tensor_fields.size());

            // Queue tensor reading tasks
            for (const auto &field : tensor_fields)
            {
                futures.push_back(
                    thread_pool->enqueue([this, field]()
                                         { return read_tensor_data_mt(field); }));
            }

            // Collect results in order
            tensors.reserve(futures.size());
            for (auto &future : futures)
            {
                tensors.push_back(future.get());
            }
        }

        ReaderTensor read_tensor_data_mt(const ReaderField &field)
        {
            GGMLQuantizationType type = static_cast<GGMLQuantizationType>(
                std::get<uint32_t>(field.values[0]));
            uint64_t tensor_offset = std::get<uint64_t>(field.values[1]);
            uint32_t n_dims = std::get<uint32_t>(field.values[2]);

            std::vector<uint64_t> dims;
            dims.reserve(n_dims);
            for (uint32_t i = 0; i < n_dims; i++)
            {
                dims.push_back(std::get<uint64_t>(field.values[3 + i]));
            }

            uint64_t n_elements = 1;
            for (uint64_t dim : dims)
            {
                n_elements *= dim;
            }

            auto quant_info = GGML_QUANT_SIZES.find(type);
            if (quant_info == GGML_QUANT_SIZES.end())
            {
                throw std::runtime_error("Unknown quantization type");
            }
            uint32_t block_size = quant_info->second.block_size;
            uint32_t type_size = quant_info->second.type_size;

            uint64_t n_bytes = (n_elements * type_size + block_size - 1) / block_size;

            // Read tensor data with mutex protection
            std::vector<uint8_t> data(n_bytes);
            {
                std::lock_guard<std::mutex> lock(file_mutex);
                uint64_t abs_offset = data_offset + tensor_offset;
                file.seekg(abs_offset);
                file.read(reinterpret_cast<char *>(data.data()), n_bytes);
            }

            return ReaderTensor(
                field.name,
                type,
                dims,
                n_elements,
                n_bytes,
                tensor_offset,
                std::move(data),
                field);
        }

        ReaderField read_tensor_info(uint64_t offset)
        {
            ReaderField field(offset, read_string());

            // Read dimensions
            uint32_t n_dims = read_value<uint32_t>();
            std::vector<uint64_t> dims(n_dims);
            for (uint32_t i = 0; i < n_dims; i++)
            {
                dims[i] = read_value<uint64_t>();
            }

            // Read type and offset
            GGMLQuantizationType type = static_cast<GGMLQuantizationType>(read_value<uint32_t>());
            uint64_t tensor_offset = read_value<uint64_t>();

            // Store values
            field.values.push_back(static_cast<uint32_t>(type));
            field.values.push_back(tensor_offset);

            // Store dimensions
            field.values.push_back(static_cast<uint32_t>(n_dims));
            for (uint64_t dim : dims)
            {
                field.values.push_back(dim);
            }

            return field;
        }

        void read_tensor_data(const ReaderField &field)
        {
            // Extract tensor information from field
            GGMLQuantizationType type = static_cast<GGMLQuantizationType>(
                std::get<uint32_t>(field.values[0]));
            uint64_t tensor_offset = std::get<uint64_t>(field.values[1]);
            uint32_t n_dims = std::get<uint32_t>(field.values[2]);

            // Get dimensions
            std::vector<uint64_t> dims;
            dims.reserve(n_dims);
            for (uint32_t i = 0; i < n_dims; i++)
            {
                dims.push_back(std::get<uint64_t>(field.values[3 + i]));
            }

            // Calculate tensor size
            uint64_t n_elements = 1;
            for (uint64_t dim : dims)
            {
                n_elements *= dim;
            }

            // Get block size and type size
            auto quant_info = GGML_QUANT_SIZES.find(type);
            if (quant_info == GGML_QUANT_SIZES.end())
            {
                throw std::runtime_error("Unknown quantization type");
            }
            uint32_t block_size = quant_info->second.block_size;
            uint32_t type_size = quant_info->second.type_size;

            // Calculate bytes needed
            uint64_t n_bytes = (n_elements * type_size + block_size - 1) / block_size;

            // Read tensor data
            std::vector<uint8_t> data(n_bytes);
            uint64_t abs_offset = data_offset + tensor_offset;
            file.seekg(abs_offset);
            file.read(reinterpret_cast<char *>(data.data()), n_bytes);

            // Create tensor
            tensors.emplace_back(
                field.name,
                type,
                dims,
                n_elements,
                n_bytes,
                tensor_offset,
                std::move(data),
                field);
        }
    };

}

int main()
{
    try
    {
        gguf::GGUFReader reader("../../models/Mistral-7b-Q4_K_M/mistral-Q4_K_M.gguf");

        // Get metadata
        auto architecture = reader.get_field("general.architecture");
        if (architecture)
        {
            std::cout << "Model architecture: "
                      << std::get<std::string>(architecture->values[0]) << "\n";
        }

        // Access tensors666666666666666
        for (size_t i = 0; i < reader.get_tensor_count(); i++)
        {
            const auto &tensor = reader.get_tensor(i);
            std::cout << "Tensor " << i << ": " << tensor.name
                      << " (elements: " << tensor.n_elements << ")\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}