package com.ai.mentor.backend.controller;

import com.ai.mentor.backend.model.ChatHistory;
import com.ai.mentor.backend.model.User;
import com.ai.mentor.backend.repository.ChatHistoryRepository;
import com.ai.mentor.backend.repository.UserRepository;
import com.ai.mentor.backend.service.ChatService;
import com.ai.mentor.backend.service.JwtService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@CrossOrigin(origins = {"http://localhost:5500", "http://127.0.0.1:5500"}, allowCredentials = "true")
@RestController
@RequestMapping("/api/chat")
public class ChatController {

    @Autowired
    private ChatService chatService;

    @Autowired
    private JwtService jwtService;

    @Autowired
    private ChatHistoryRepository chatHistoryRepository;

    @Autowired
    private UserRepository userRepository;

    /**
     * Stream chat response (with optional authentication)
     */
    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> getChatResponse(
            @RequestBody ChatRequest request,
            @RequestHeader(value = "Authorization", required = false) String authHeader) {

        String email = null;
        if (authHeader != null && authHeader.startsWith("Bearer ")) {
            try {
                String token = authHeader.substring(7);
                email = jwtService.extractUsername(token); // Extracts email from JWT
                System.out.println("✅ Authenticated user: " + email);
            } catch (Exception e) {
                System.err.println("⚠️ Invalid token, continuing without auth");
            }
        }

        return chatService.getChatResponseStream(request.getUserInput(), request.getRole(), email);
    }

    /**
     * Get chat history for authenticated user
     */
    @GetMapping("/history")
    public ResponseEntity<?> getChatHistory(@RequestHeader(value = "Authorization", required = false) String authHeader) {
        System.out.println("📋 History request received");

        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            System.err.println("❌ No valid authorization header");
            return ResponseEntity.status(401).body("No authentication token provided");
        }

        try {
            String token = authHeader.substring(7);
            String email = jwtService.extractUsername(token); // This actually extracts email
            System.out.println("✅ Loading history for user: " + email);

            // Find by email instead of username
            User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found: " + email));

            List<ChatHistory> history = chatHistoryRepository.findByUser(user);
            System.out.println("📦 Found " + history.size() + " chat messages");

            // Convert to DTO
            List<ChatHistoryDTO> dtoList = history.stream()
                    .map(chat -> new ChatHistoryDTO(
                            chat.getId(),
                            chat.getMessage(),
                            chat.getResponse(),
                            chat.getTimestamp().toString()
                    ))
                    .collect(Collectors.toList());

            return ResponseEntity.ok(dtoList);

        } catch (Exception e) {
            System.err.println("❌ Error loading history: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(500).body("Error loading history: " + e.getMessage());
        }
    }

    /**
     * TEST ENDPOINT: Manually save a test chat message
     */
    @PostMapping("/test-save")
    public ResponseEntity<?> testSave(@RequestHeader("Authorization") String authHeader) {
        try {
            String token = authHeader.replace("Bearer ", "");
            String email = jwtService.extractUsername(token);

            User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found"));

            ChatHistory testChat = new ChatHistory("Test message", "Test response", user);
            chatHistoryRepository.save(testChat);

            System.out.println("✅ Test chat saved successfully!");
            return ResponseEntity.ok("Test chat saved!");

        } catch (Exception e) {
            System.err.println("❌ Failed to save test: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(500).body("Error: " + e.getMessage());
        }
    }

    // Request DTO
    public static class ChatRequest {
        private String userInput;
        private String role;

        public ChatRequest() {}

        public String getUserInput() { return userInput; }
        public void setUserInput(String userInput) { this.userInput = userInput; }

        public String getRole() { return role; }
        public void setRole(String role) { this.role = role; }
    }

    // Response DTO for history
    public static class ChatHistoryDTO {
        private Long id;
        private String message;
        private String response;
        private String timestamp;

        public ChatHistoryDTO(Long id, String message, String response, String timestamp) {
            this.id = id;
            this.message = message;
            this.response = response;
            this.timestamp = timestamp;
        }

        public Long getId() { return id; }
        public String getMessage() { return message; }
        public String getResponse() { return response; }
        public String getTimestamp() { return timestamp; }
    }
}