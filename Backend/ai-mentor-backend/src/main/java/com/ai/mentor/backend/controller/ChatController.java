package com.ai.mentor.backend.controller;

import com.ai.mentor.backend.model.ChatHistory;
import com.ai.mentor.backend.model.User;
import com.ai.mentor.backend.repository.ChatHistoryRepository;
import com.ai.mentor.backend.repository.UserRepository;
import com.ai.mentor.backend.service.JwtService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@CrossOrigin(origins = {"http://localhost:5500", "http://127.0.0.1:5500"}, allowCredentials = "true")
@RestController
@RequestMapping("/api/chat")
public class ChatController {

    @Autowired
    private JwtService jwtService;

    @Autowired
    private ChatHistoryRepository chatHistoryRepository;

    @Autowired
    private UserRepository userRepository;

    private final RestTemplate restTemplate = new RestTemplate();
    private final String PYTHON_API_URL = "http://127.0.0.1:8000/chat";

    /**
     * Non-streaming chat endpoint
     */
    @PostMapping(value = "/message")
    public ResponseEntity<?> getChatResponse(
            @RequestBody ChatRequest request,
            @RequestHeader(value = "Authorization", required = false) String authHeader) {

        try {
            // Validate input
            if (request.getUserInput() == null || request.getUserInput().trim().isEmpty()) {
                return ResponseEntity.badRequest().body(Map.of("error", "User input is required"));
            }

            // Extract user email if authenticated
            String email = null;
            User user = null;
            if (authHeader != null && authHeader.startsWith("Bearer ")) {
                try {
                    String token = authHeader.substring(7);
                    email = jwtService.extractUsername(token);
                    user = userRepository.findByEmail(email).orElse(null);
                    System.out.println("✅ Authenticated user: " + email);
                } catch (Exception e) {
                    System.err.println("⚠️ Invalid token, continuing without auth");
                }
            }

            // Call Python microservice
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            Map<String, String> body = new HashMap<>();
            body.put("userInput", request.getUserInput());
            body.put("role", request.getRole() != null ? request.getRole() : "Career_mentor");

            HttpEntity<Map<String, String>> entity = new HttpEntity<>(body, headers);

            ResponseEntity<Map> response = restTemplate.postForEntity(
                    PYTHON_API_URL,
                    entity,
                    Map.class
            );

            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                String botResponse = (String) response.getBody().get("response");

                // Save to database if user is authenticated
                if (user != null && botResponse != null) {
                    try {
                        ChatHistory chatHistory = new ChatHistory(
                                request.getUserInput(),
                                botResponse,
                                user
                        );
                        chatHistoryRepository.save(chatHistory);
                        System.out.println("💾 Chat saved to history for user: " + email);
                    } catch (Exception e) {
                        System.err.println("⚠️ Failed to save chat history: " + e.getMessage());
                    }
                }

                return ResponseEntity.ok(response.getBody());
            } else {
                return ResponseEntity.status(500)
                        .body(Map.of("error", "Failed to get response from AI service"));
            }

        } catch (Exception e) {
            System.err.println("❌ Error in chat endpoint: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(500)
                    .body(Map.of("error", "Error: " + e.getMessage()));
        }
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
            String email = jwtService.extractUsername(token);
            System.out.println("✅ Loading history for user: " + email);

            User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found: " + email));

            List<ChatHistory> history = chatHistoryRepository.findByUser(user);
            System.out.println("📦 Found " + history.size() + " chat messages");

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

    @GetMapping("/health")
    public ResponseEntity<?> health() {
        try {
            ResponseEntity<String> response = restTemplate.getForEntity(
                    "http://127.0.0.1:8000/health",
                    String.class
            );

            if (response.getStatusCode().is2xxSuccessful()) {
                return ResponseEntity.ok(Map.of(
                        "status", "healthy",
                        "pythonService", "connected",
                        "mode", "non-streaming"
                ));
            }
        } catch (Exception e) {
            return ResponseEntity.status(503)
                    .body(Map.of(
                            "status", "unhealthy",
                            "error", "Python microservice not reachable"
                    ));
        }

        return ResponseEntity.status(503)
                .body(Map.of("status", "unhealthy"));
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