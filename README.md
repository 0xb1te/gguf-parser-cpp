# Runnning

You will need to download a model for testing, place it on your desired folder and change the route on the line (absolute or relative route, I only tested with relative)

```bash
gguf::GGUFReader reader("../../models/Mistral-7b-Q4_K_M/mistral-Q4_K_M.gguf");
```

After this, you will be able to compile by doing

```bash
chmod +x build.sh
./build.sh
```

After that, the info of the .gguf will be shown on the console, counting and showing info of the metadata, tensors... as said on the
format (specification: [a link](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md))

I tried and debugged with the following model: [a link](https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF)
Downloaded at: [a link](https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/blob/main/mixtral-8x7b-v0.1.Q4_K_M.gguf)

So far so good, it works!
