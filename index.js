const dotenv = require("dotenv");
const path = require("path");
dotenv.config({ path: path.join(__dirname, "./.env") });

const {
  queryEmbeddings,
  processDocument,
  upsertEmbeddings,
} = require("./pinecone");
const { default: axios } = require("axios");

const runRAGPipeline = async (query, nameSpace) => {
  //Create a embedding from the query
  const queryEmbedding = await axios.post(
    "https://api.openai.com/v1/embeddings",
    {
      model: "text-embedding-ada-002",
      input: query,
    },
    {
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
    }
  );

  // Extract the embedding from the object
  const queryVector = queryEmbedding.data.data[0].embedding;

  // Retrieve relevant document chunks from Pinecone
  const retrievedChunks = await queryEmbeddings(queryVector, nameSpace);

  console.log("Retrieved Chunks:", retrievedChunks);

  // Combine retrieved chunks and send them to ChatGPT for final answer
  const context = retrievedChunks
    .map((chunk) => chunk.metadata.text)
    .join("\n");

  const chatGPTResponse = await axios.post(
    "https://api.openai.com/v1/chat/completions",
    {
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content:
            "You are a helpful assistant. Do not answer anything that is out of context.",
        },
        {
          role: "user",
          content: `Based on the following context: ${context}. Answer the question: ${query}`,
        },
      ],
    },
    {
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
    }
  );
  const out = chatGPTResponse.data.choices[0].message.content;
  console.log("ChatGPT Response:", out);
  return out;
};

const nameSpace = "new";

const saveDataToPinecone = async (document) => {
  const embeddingChunks = await processDocument(document);
  await upsertEmbeddings(embeddingChunks, nameSpace);
  console.log("Data is saved in pinecone db");
};

const docs = `RAG offers several advantages augmenting traditional methods of text generation, especially when dealing with factual information or data-driven responses. Here are some key reasons why using RAG can be beneficial: Access to fresh information
LLMs are limited to their pre-trained data. This leads to outdated and potentially inaccurate responses. RAG overcomes this by providing up-to-date information to LLMs.
Factual grounding
LLMs are powerful tools for generating creative and engaging text, but they can sometimes struggle with factual accuracy. This is because LLMs are trained on massive amounts of text data, which may contain inaccuracies or biases.
Providing “facts” to the LLM as part of the input prompt can mitigate “gen AI hallucinations.” The crux of this approach is ensuring that the most relevant facts are provided to the LLM, and that the LLM output is entirely grounded on those facts while also answering the user’s question and adhering to system instructions and safety constraints.
Using Gemini’s long context window (LCW) is a great way to provide source materials to the LLM. If you need to provide more information than fits into the LCW, or if you need to scale up performance, you can use a RAG approach that will reduce the number of tokens, saving you time and cost.`;

// saveDataToPinecone(docs);

runRAGPipeline("What is rag ?", nameSpace);
