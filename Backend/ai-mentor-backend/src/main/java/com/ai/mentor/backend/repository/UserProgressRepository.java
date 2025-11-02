package com.ai.mentor.backend.repository;

import com.ai.mentor.backend.model.User;
import com.ai.mentor.backend.model.UserProgress;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserProgressRepository extends JpaRepository<UserProgress, Long> {
    Optional<UserProgress> findByUser(User user);
}