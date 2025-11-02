package com.ai.mentor.backend.service;

import com.ai.mentor.backend.model.ChatHistory;
import com.ai.mentor.backend.model.User;
import com.ai.mentor.backend.repository.ChatHistoryRepository;
import com.ai.mentor.backend.repository.UserRepository;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class ChatHistoryService {

    private final ChatHistoryRepository chatHistoryRepository;
    private final UserRepository userRepository;
    private final JwtService jwtService;

    public ChatHistoryService(ChatHistoryRepository chatHistoryRepository,
                              UserRepository userRepository,
                              JwtService jwtService) {
        this.chatHistoryRepository = chatHistoryRepository;
        this.userRepository = userRepository;
        this.jwtService = jwtService;
    }

    public void saveChat(String token, String message, String response) {
        String username = jwtService.extractUsername(token);
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));

        ChatHistory chat = new ChatHistory(message, response, user);
        chatHistoryRepository.save(chat);
    }

    public List<ChatHistory> getChatHistory(String token) {
        String username = jwtService.extractUsername(token);
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));

        return chatHistoryRepository.findByUser(user);
    }
}
