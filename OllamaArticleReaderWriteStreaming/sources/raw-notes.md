# Raw notes — running LLMs locally (unedited brain dump)

- ollama makes it easy, one command to pull a model, `ollama run gemma`
- privacy: data never leaves my machine. big deal for legal/medical/confidential stuff
- no per-token cost. cloud APIs bill per token, adds up fast for high volume
- works offline, on a plane, no internet needed
- latency: no network round trip, feels snappy for short prompts
- downside: needs decent hardware, GPU or lots of RAM. big models are slow on CPU
- can fine-tune / customize on your own data
- good for experimenting without a credit card
- tool calling works locally too (this whole project!)
- model quality: local models smaller than frontier cloud models, tradeoff
