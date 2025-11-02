package com.ai.mentor.backend.repository;

import com.ai.mentor.backend.model.RoadmapProgress;
import com.ai.mentor.backend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface RoadmapProgressRepository extends JpaRepository<RoadmapProgress, Long> {
    List<RoadmapProgress> findByUser(User user);
    Optional<RoadmapProgress> findByUserAndRoadmapIdAndPhaseNumber(User user, Integer roadmapId, Integer phaseNumber);
}