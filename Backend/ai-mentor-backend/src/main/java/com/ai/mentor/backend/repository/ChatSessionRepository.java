package com.ai.mentor.backend.repository;


import com.ai.mentor.backend.model.ChatSession;
import com.ai.mentor.backend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ChatSessionRepository extends JpaRepository<ChatSession, Long> {
    List<ChatSession> findByUser(User user);
}

