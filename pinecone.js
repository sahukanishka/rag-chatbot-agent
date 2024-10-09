const { Pinecone } = require("@pinecone-database/pinecone");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { default: axios } = require("axios");

//Create a client
const client = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

// Set your index name
const indexName = "demo-index";

const index = client.index(indexName);

// Function to upsert document embeddings
const upsertEmbeddings = async (embeddings, nameSpace) => {
  if (nameSpace) await index.namespace(nameSpace).upsert(embeddings);
  else await index.upsert(embeddings);
};

// Function to query Pinecone for relevant embeddings
const queryEmbeddings = async (queryEmbedding, nameSpace) => {
  const result = await index.namespace(nameSpace).query({
    vector: queryEmbedding,
    topK: 3,
    includeMetadata: true,
  });
  return result.matches;
};

const chunkDocument = async (document, chunkSize = 500, chunkOverlap = 50) => {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize, // Size of each chunk in characters
    chunkOverlap, // Overlap between chunks to preserve context
  });
  const output = await splitter.createDocuments([document]);
  return output;
};

// Function to generate embeddings using OpenAI API
const getEmbedding = async (text) => {
  const response = await axios.post(
    `${process.env.OPENAI_BASE_URL}/v1/embeddings`,
    {
      model: "text-embedding-ada-002",
      input: text,
    },
    {
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
    }
  );
  return response.data.data[0].embedding;
};

// Function to process a document: chunk and embed
const processDocument = async (document) => {
  const chunks = await chunkDocument(document);
  const embeddings = await Promise.all(
    chunks.map(async (chunk, index) => ({
      id: `chunk-${index}`,
      values: await getEmbedding(chunk.pageContent),
      metadata: { text: chunk.pageContent },
    }))
  );
  return embeddings;
};

//export the functions
module.exports = { upsertEmbeddings, queryEmbeddings, processDocument };
