package com.ai.mentor.backend.repository;

import com.ai.mentor.backend.model.Message;
import com.ai.mentor.backend.model.ChatSession;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface MessageRepository extends JpaRepository<Message, Long> {
    List<Message> findByChatSessionOrderByTimestampAsc(ChatSession chatSession);
}

