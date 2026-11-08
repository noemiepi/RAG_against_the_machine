<div align="center">
    <i>This project has been created as part of the 42 curriculum by npillet</i>
    <h1>RAG against the machine</h1>
    <h3>Will you answer my questions?</h3>
</div>

## Table of content
- [Description](#description)
- [Instructions](#instructions)
    - [Example Usage](#example-usage)
- [System Architecture](#system-architecture)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval Method](#retrieval-method)
- [Performance Analysis](#performance-analysis)
- [Design Decisions](#design-decisions)
- [Challenges Faced](#challenges-faced)
- [Resources](#resources)

## Description
A model has only the informations that were given during its training. In order to add new knowledge, instead of retraining it with the new data, we can can use **Retrieval-Augmented Generation** also known as **RAG**.</br>
With this, you give access to external sources of the wanted knowledge and let the model pull an answer from this.

RAG has four stages:
- **Indexing**: organise the data so it can be searched
- **Retrieving**: match a question agains the index and pull the most relevant snippets
- **Augmenting**: filter those snippets and place them in a model's context window
- **Generating**:read thet context and produce the answer

For this project, the goal is to recreate a working **RAG**.

## Instructions
With this list of commands, once entered inside the terminal and after installing the dependencies, the full pipeline will be able to run.
``` bash
# Create the index
make index

# Retrieve the sources for a query
make search

# Retrieve the sources for a dataset
make search_dataset

# Answer a query with the previously retrieved sources
make answer

# Answer a dataset with the previously retrieved sources
make answer_dataset

# Measure the retrieval quality with a recall@k metric
make evaluate
```

And below, you will find other commands:
| Command | Description |
| :---: | --- |
| `make install` | Install the project's dependencies |
| `make run` | Execute the program (like the `make` command) |
| `make debug` | Run the script using the Python built-in debugger |
| `make clean` | Remove temporary files and caches |
| `make lint` | Execute the `flake8` and `mypy` commands |
| `make lint-strict` | Execute the `flake8` and a stricter version of `mypy` commands |

### Example Usage
By entering these commands in order, the full pipeline will be able to run.
``` bash
# Create the index
uv run python -m src index --max_chunk_size 2000

# Retrieve the sources for a dataset
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

# Measure the retrieval quality with a recall@k metric
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000

# Answer a dataset with the previously retrieved sources
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## System Architecture
Below is a visual representation of a classic RAG pipeline:
<div align="center">
  <img src="assets/pipeline.png" />
</div>

In my implementation

*Describe your RAG pipeline components and how they interact*

```mermaid
flowchart LR

    classDef indexmethod fill:#cad182,stroke:#606c38,stroke-width:2px,color:#333;
    classDef index fill:#deebf7,stroke:#c6dbef,stroke-width:2px,color:#333;
    classDef retrieve fill:#fec89a,stroke:#dda15e,stroke-width:2px,color:#333;
    classDef generate fill:#f69697,stroke:#ee6b6e,stroke-width:2px,color:#333;
    classDef evaluate fill:#f3e5f5,stroke:#e1bee7,stroke-width:2px,color:#333;


    subgraph Indexing
    A(Corpus - data/raw/vllm-0.10.1) --> B(Chunking - code/markdown)
    end

    B --> C(Index BM25 / TF-IDF - data/processed)
    C --> D(Retrieve top-k sources)

    subgraph Retrieval and Generation
    D --> E(Augment - build context)
    E --> G(Generate - Qwen3-0.6B)
    G --> H(Answer - JSON)
    end

    D --> |scoring| F(Evaluate recall@k - IoU > 0.05)

    class A,B index
    class C indexmethod
    class D,E retrieve
    class G,H generate
    class F evaluate

    linkStyle default stroke:gray;
```

## Chunking Strategy
During the chunking, the file format needs to be considered. A code won't be seperated to same way as a text. In this project, we will face three formats:
- **Python code(.py)**:

- **Documentation(.md and .txt)**:

- **Overlap?**:

*Explain your approach to document segmentation*

## Retrieval Method
bm25s
stemmer

*Detail the retrieval algorithm and ranking mechanism*

## Performance Analysis
There are a few expected performances, such as:
- **Indexing time**: at most 5 minutes for the whole corpus
- **Retrieval throughput**: at most 90 seconds for 200 questions
- **Recall@5**: at least 80% on docs questions and 50% on code questions

For the indexing time and the retrieval throughput, the program manage to do each in just a few seconds.

For the recall@5, below are the results:</br>
- This is the documents recall@k:
<div align="center">
  <img src="assets/doc_recall@k.png" />
</div>
The recall@5 is equal to 84.0%, which is above 80.0% and therefore passes.

</br>

- This is the codes recall@k:
<div align="center">
  <img src="assets/code_recall@k.png" />
</div>
The recall@5 is equal to 54.5%, which is above 50.0% and therefore also passes.

## Design Decisions
BM25 over TF-IDF 
*Explain key implementation choices*

## Challenges Faced
*Document difficulties encountered and solutions*

## Resources
### Notions
- [Introduction to RAG](https://blog.stephane-robert.info/docs/developper/programmation/python/rag-introduction/)

- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

#### Dependencies
- [Python Fire](https://github.com/google/python-fire)

- [LangChain](https://www.ibm.com/fr-fr/think/topics/langchain)

- [LangChain Text Splitter](https://docs.langchain.com/oss/python/integrations/splitters)

- [BM25](https://github.com/xhluca/bm25s)

### GitHub
- [Overtekk](https://github.com/Overtekk/RAG-against-the-machine)
