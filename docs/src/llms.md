---
title: llms.txt
sidebar_position: 5
---

# llms.txt

Below you can find the Modlib documentation files in the [llms.txt](https://llmstxt.org/) format. This allows large language models (LLMs) and agents to access programming documentation and APIs, particularly useful within integrated development environments (IDEs).

| Modlib llms.txt | |
|---------------------------|--------------------------|
| Web                       | Available soon!          |
| Source                    | [https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library/blob/main/docs/llms.txt](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library/blob/main/docs/llms.txt)  |


:::warning  
Even with access to up-to-date documentation, current state-of-the-art models may not always generate correct code. Treat the generated code as a starting point, and always review it before shipping code to production.
:::

## Using llms.txt file

The Modlib llms.txt file typically contains several thousand tokens, exceeding the context window limitations of some LLMs. To effectively use this file:

1. **With IDEs (e.g., Cursor, Windsurf):**

    Add the llms.txt as custom documentation. The IDE will automatically chunk and index the content, implementing Retrieval-Augmented Generation (RAG).

2. **Without IDE support:**

    Use a chat model with a large context window.
    Implement a RAG strategy to manage and query the documentation efficiently.