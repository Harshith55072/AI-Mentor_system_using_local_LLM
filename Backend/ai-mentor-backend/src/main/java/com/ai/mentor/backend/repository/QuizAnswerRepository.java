package com.ai.mentor.backend.repository;

import com.ai.mentor.backend.model.QuizAnswer;
import com.ai.mentor.backend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface QuizAnswerRepository extends JpaRepository<QuizAnswer, Long> {
    List<QuizAnswer> findByUser(User user);
    Optional<QuizAnswer> findByUserAndRoadmapIdAndPhaseNumber(User user, Integer roadmapId, Integer phaseNumber);
    List<QuizAnswer> findByUserAndPassed(User user, Boolean passed);
}