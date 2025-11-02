package com.ai.mentor.backend.service;

import com.ai.mentor.backend.model.ChatHistory;
import com.ai.mentor.backend.model.User;
import com.ai.mentor.backend.repository.ChatHistoryRepository;
import com.ai.mentor.backend.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import java.util.concurrent.atomic.AtomicReference;

@Service
public class ChatService {

    private final WebClient webClient;

    @Autowired
    private ChatHistoryRepository chatHistoryRepository;

    @Autowired
    private UserRepository userRepository;

    public ChatService() {
        this.webClient = WebClient.builder()
                .baseUrl("http://127.0.0.1:8000")
                .build();
    }

    /**
     * Streams response from FastAPI and saves to database if user is authenticated
     */
    public Flux<String> getChatResponseStream(String userInput, String role, String email) {
        ChatRequest payload = new ChatRequest(userInput, role);

        // Use AtomicReference for thread-safe accumulation in reactive streams
        AtomicReference<String> fullResponse = new AtomicReference<>("");

        System.out.println("🚀 Starting chat stream for: " + (email != null ? email : "guest"));
        System.out.println("📝 User message: " + userInput);

        return webClient.post()
                .uri("/chat/stream")
                .contentType(MediaType.APPLICATION_JSON)
                .accept(MediaType.TEXT_EVENT_STREAM)
                .body(BodyInserters.fromValue(payload))
                .retrieve()
                .bodyToFlux(String.class)
                .doOnNext(chunk -> {
                    // Debug: Log every chunk received
                    System.out.println("📦 RAW CHUNK: [" + chunk + "]");
                })
                .map(chunk -> {
                    if (chunk.startsWith("data: ")) {
                        String content = chunk.substring(6).trim();

                        System.out.println("✂️ EXTRACTED CONTENT: [" + content + "]");

                        if (content.equals("[DONE]")) {
                            System.out.println("🏁 Received [DONE] signal");
                            return ""; // Filter will remove this
                        }

                        // Accumulate response
                        fullResponse.updateAndGet(current -> {
                            String updated = current + content;
                            System.out.println("📝 ACCUMULATED SO FAR (" + updated.length() + " chars): " +
                                    updated.substring(0, Math.min(50, updated.length())) + "...");
                            return updated;
                        });
                        return "data: " + content + "\n\n";
                    } else {
                        System.out.println("⚠️ CHUNK DOESN'T START WITH 'data:' - passing through");
                        // Accumulate even if it doesn't start with "data:"
                        fullResponse.updateAndGet(current -> current + chunk);
                        return chunk;
                    }
                })
                .filter(s -> !s.isEmpty())
                .doOnComplete(() -> {
                    // Save when stream completes
                    String response = fullResponse.get();
                    System.out.println("✅ Stream completed! Response length: " + response.length());

                    if (email != null && !response.isEmpty()) {
                        System.out.println("💾 Saving chat to database...");
                        saveToHistory(email, userInput, response);
                    } else {
                        System.out.println("⚠️ Not saving: email=" + (email != null ? email : "null") +
                                ", response length=" + response.length());
                    }
                })
                .doOnError(error -> {
                    System.err.println("❌ FastAPI streaming error: " + error.getMessage());
                    error.printStackTrace();
                })
                .onErrorResume(error -> {
                    return Flux.just("data: ⚠️ AI service unavailable\n\n");
                });
    }

    /**
     * Save chat exchange to database
     */
    private void saveToHistory(String email, String userMessage, String aiResponse) {
        try {
            System.out.println("🔍 Looking up user: " + email);
            User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found: " + email));

            System.out.println("📦 Creating ChatHistory object...");
            ChatHistory chatHistory = new ChatHistory(userMessage, aiResponse, user);

            System.out.println("💿 Saving to database...");
            chatHistoryRepository.save(chatHistory);

            System.out.println("✅ Successfully saved chat for user: " + email);
        } catch (Exception e) {
            System.err.println("❌ Failed to save chat history: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // DTO for FastAPI request
    private static class ChatRequest {
        private final String userInput;
        private final String role;

        public ChatRequest(String userInput, String role) {
            this.userInput = userInput;
            this.role = role;
        }

        public String getUserInput() { return userInput; }
        public String getRole() { return role; }
    }
}