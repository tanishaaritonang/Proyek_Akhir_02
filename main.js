import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { retriever } from "./retriever.js";
import { combineDocuments } from "./combineDocuments.js";
import { formatConvHistory } from "./formatConvHistory.js";
import { createClient } from "@supabase/supabase-js";
import { supabase } from "./db/db.js";

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;
const openAIApiKey = process.env.OPENAI_API_KEY;

const llm = new ChatOpenAI({ openAIApiKey });
const client = createClient(supabaseUrl, supabaseKey);
export default client;
const convHistory = new Map();

const standaloneQuestionTemplate = `Given some conversation history (if any) and a question, convert the question to a standalone question. 

Conversation history: {conv_history}
Question: {question}

Standalone question:`;

const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
  standaloneQuestionTemplate
);


const answerTemplate = `You are a helpful, child-friendly, and enthusiastic support bot.
Answer questions creatively using the given context in simple Indonesian, wrapped in a short narrative (max 4 sentences).
Always follow this structure for crafting your answer:

Start with a simple analogy
Give a straightforward facts explanation.
Briefly explain the consequence.
End with positive, uplifting words!
Do not use emojis.

Context: {context}
Conversation History: {conv_history}
Question: {question}

Answer:
`



const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);

const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser());

const standaloneQuestionChain = standaloneQuestionPrompt
  .pipe(llm)
  .pipe(new StringOutputParser());

const retrieverChain = RunnableSequence.from([
  (prevResult) => prevResult.standalone_question,
  retriever,
  combineDocuments,
]); //making context

const chain = RunnableSequence.from([
  {
    standalone_question: standaloneQuestionChain,
    original_input: new RunnablePassthrough(),
  },
  {
    context: retrieverChain,
    question: ({ original_input }) => original_input.question,
    conv_history: ({ original_input }) => original_input.conv_history,
    
  },
  answerChain,
]);

                        


//////
export async function progressConversation(question, sessionId, userId) {
  try {
    if (!convHistory.has(sessionId)) {
      convHistory.set(sessionId, []);
    }
    const sessionHistory = convHistory.get(sessionId);

    // Waktu saat user mengirim pertanyaan
    const questionTime = new Date();

    // 1. Simpan pertanyaan dulu supaya waktu aslinya tercatat
    const { error: questionInsertError } = await supabase
      .from("messages")
      .insert([
        {
          session_id: sessionId,
          message_type: "question",
          body: question,
          created_at: questionTime.toISOString(),
        },
      ]);

    if (questionInsertError) {
      console.error("Error storing question:", questionInsertError);
    }

    // Pastikan session ada di tabel
    const { data: existingSession, error: sessionError } = await supabase
      .from("sessions")
      .select("id")
      .eq("id", sessionId)
      .single();

    if (sessionError && !existingSession) {
      const { error: createError } = await supabase.from("sessions").insert([
        {
          id: sessionId,
          created_at: questionTime.toISOString(),
          user_id: userId,
        },
      ]);

      if (createError) {
        console.error("Error creating session:", createError);
      }
    }

    // 2. Jalankan LLM (ini bagian yang memakan waktu, misalnya 9 detik)
    const response = await chain.invoke({
      question: question,
      conv_history: formatConvHistory(sessionHistory),
    });

    // Hitung waktu setelah jawaban selesai
    const responseTime = new Date();
    const responseDurationMs = responseTime.getTime() - questionTime.getTime();

    // Update conversation history di memory
    sessionHistory.push(question);
    sessionHistory.push(response);
    convHistory.set(sessionId, sessionHistory);

    // 3. Simpan jawaban ke DB terpisah
    const { error: responseInsertError } = await supabase
      .from("messages")
      .insert([
        {
          session_id: sessionId,
          message_type: "response",
          body: response,
          created_at: responseTime.toISOString(),
          response_duration_ms: responseDurationMs, // kolom tambahan
        },
      ]);

    if (responseInsertError) {
      console.error("Error storing response:", responseInsertError);
    }

    // Deteksi apakah input berupa pertanyaan
    const isQuestion =
      /^(what|who|when|where|why|how|is|are|can|could|would|will|do|does|did|have|has|may|might)\b/i.test(
        question
      ) || question.trim().endsWith("?");

    // Simpan embedding jika ini pertanyaan
    if (isQuestion) {
      try {
        const embeddingResponse = await fetch(
          "https://api.openai.com/v1/embeddings",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${openAIApiKey}`,
            },
            body: JSON.stringify({
              input: question,
              model: "text-embedding-3-small",
            }),
          }
        );

        const embeddingData = await embeddingResponse.json();
        const embedding = embeddingData.data[0].embedding;

        Promise.all([
          getSimilarPopularPrompts(question, true, embedding),
          storeUserPrompt(question, embedding),
        ]).catch((error) => {
          console.error("Error in background tasks:", error);
        });
      } catch (error) {
        console.error("Error tracking prompt:", error);
      }
    }

    // Kembalikan jawaban ke user
    return response;
  } catch (error) {
    console.error("Error in conversation:", error);
    return "I'm sorry, I encountered an error. Please try again or contact support.";
  }
}


async function getSimilarPopularPrompts(
  question,
  incrementSimilar = false,
  embedding
) {
  try {
    // Find similar prompts
    const { data, error } = await client.rpc("find_similar_prompts", {
      query_embedding: embedding,
      similarity_threshold: 0.6,
      match_count: 5, // Get more to have better options after filtering
    });

    if (error) throw error;

    // Filter out the exact match and low similarity
    const filtered = data.filter(
      (item) =>
        item.similarity > 0.7 &&
        item.prompt.toLowerCase() !== question.toLowerCase()
    );

    // console.log("Filtered similar prompts:", filtered); PENTING

    // If we're incrementing similar prompts (for when a question is asked)
    if (incrementSimilar && filtered.length > 0) {
      try {
        // Calculate increment based on similarity
        const updatesPromises = data.map((item) => {
          // Higher similarity = higher increment
          return client
            .from("user_prompts")
            .update({
              count: item.count + 1,
              last_used_at: new Date().toISOString(),
            })
            .eq("id", item.id);
        });

        await Promise.all(updatesPromises);
      } catch (error) {
        console.error("Error incrementing similar prompts:", error);
      }
    }

    // Return top 3 most popular similar prompts
    return filtered
      .sort((a, b) => b.count - a.count)
      .slice(0, 3)
      .map((item) => item.prompt);
  } catch (error) {
    console.error("Error finding similar prompts:", error);
    return [];
  }
}

async function storeUserPrompt(question, embedding) {
  try {
    // Check if prompt already exists
    const { data: existingPrompts, error: queryError } = await client
      .from("user_prompts")
      .select("id, count")
      .eq("prompt", question)
      .limit(1);

    if (queryError) throw queryError;

    if (existingPrompts && existingPrompts.length > 0) {
      // Update existing prompt
      console.log(`Updating existing prompt: ${question}`);

      // UPDATE user_prompts SET count = count + 1 WHERE id = count
      const { error: updateError } = await client
        .from("user_prompts")
        .update({
          count: existingPrompts[0].count + 1,
          last_used_at: new Date().toISOString(),
        })
        .eq("id", existingPrompts[0].id);

      if (updateError) throw updateError;
    } else {
      // Insert new prompt
      console.log(`Inserting new prompt: ${question}`);

      // INSERT INTO user_prompts (prompt, count, embedding) VALUES (?, ?, ?)
      const { error: insertError } = await client.from("user_prompts").insert({
        prompt: question,
        count: 1,
        embedding: embedding,
        last_used_at: new Date().toISOString(),
      });

      if (insertError) throw insertError;
    }

    return true;
  } catch (error) {
    console.error("Error storing prompt:", error);
    return false;
  }
}





