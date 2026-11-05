using CommunityToolkit.VectorData.InMemory;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using OpenAI;
using System.ClientModel;
using Vector_Search;

// 1. Get credentials from user secrets or environment variables
IConfigurationRoot config = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

var apiKey = config["OpenAI:ApiKey"] ?? Environment.GetEnvironmentVariable("OPENAI_API_KEY")
    ?? throw new InvalidOperationException("Missing configuration for OpenAI API key.");

// Standard client targeting https://api.openai.com/v1 by default
var openAiClient = new OpenAIClient(new ApiKeyCredential(apiKey));

// 2. Chat client using an OpenAI chat model (e.g., gpt-4o-mini)
IChatClient chatClient = openAiClient.GetChatClient("gpt-4o-mini").AsIChatClient();

// 3. Embedding generator using OpenAI's text-embedding-3-small
IEmbeddingGenerator<string, Embedding<float>> generator = openAiClient.GetEmbeddingClient("text-embedding-3-small").AsIEmbeddingGenerator();

// 4. Create and populate the in-memory vector store
var vectorStore = new InMemoryVectorStore();
var moviesStore = vectorStore.GetCollection<int, Movie>("movies");
await moviesStore.EnsureCollectionExistsAsync();

// 5. Ingest data
foreach (var movie in MovieData.Movies)
{
    // Generate the embedding vector using OpenAI
    movie.Vector = await generator.GenerateVectorAsync(movie.Description);

    // Upsert into vector store
    await moviesStore.UpsertAsync(movie);
}

// generate the embedding vector for the user's prompt
//var query = "I want to see family friendly movie";
var query = "A science fiction movie about space travel";
var queryEmbedding = await generator.GenerateVectorAsync(query);

// search the knowledge store based on the user's prompt
var searchResults = moviesStore.SearchAsync(queryEmbedding, top: 2);

// see the results just so we know what they look like
await foreach (var result in searchResults)
{
    Console.WriteLine($"Title: {result.Record.Title}");
    Console.WriteLine($"Description: {result.Record.Description}");
    Console.WriteLine($"Score: {result.Score}");
    Console.WriteLine();
}