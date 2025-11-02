package com.ai.mentor.backend.repository;

import com.ai.mentor.backend.model.ChatHistory;
import com.ai.mentor.backend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface ChatHistoryRepository extends JpaRepository<ChatHistory, Long> {
    List<ChatHistory> findByUser(User user);
}
