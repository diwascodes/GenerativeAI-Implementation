using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;
using OpenAI;
using System.ClientModel;
using System.Numerics.Tensors;

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


//var embedding = await generator.GenerateVectorAsync("1");
//Console.WriteLine($"Embedding vector length: {embedding.Span.Length}");
//foreach (var item in embedding.Span)
//{
//    Console.WriteLine(item);
//}

var catVector = await generator.GenerateVectorAsync("cat");
var dogVector = await generator.GenerateVectorAsync("dog");
var kittendVector = await generator.GenerateVectorAsync("kitten");
Console.WriteLine($"Cosine similarity between cat and dog: {TensorPrimitives.CosineSimilarity(catVector.Span, dogVector.Span)}");
Console.WriteLine($"Cosine similarity between cat and kitten: {TensorPrimitives.CosineSimilarity(catVector.Span, kittendVector.Span)}");
Console.WriteLine($"Cosine similarity between dog and kitten: {TensorPrimitives.CosineSimilarity(dogVector.Span, kittendVector.Span)}");

