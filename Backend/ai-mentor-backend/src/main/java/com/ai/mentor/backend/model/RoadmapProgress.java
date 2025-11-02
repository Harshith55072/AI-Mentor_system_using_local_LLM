package com.ai.mentor.backend.model;

import jakarta.persistence.*;

@Entity
@Table(name = "roadmap_progress")
public class RoadmapProgress {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "roadmap_id", nullable = false)
    private Integer roadmapId; // 1=Software Dev, 2=Data Scientist, 3=AI/ML

    @Column(name = "phase_number", nullable = false)
    private Integer phaseNumber; // 1-5

    @Column(name = "status", nullable = false)
    private String status; // "NOT_STARTED", "IN_PROGRESS", "COMPLETED"

    public RoadmapProgress() {}

    public RoadmapProgress(User user, Integer roadmapId, Integer phaseNumber, String status) {
        this.user = user;
        this.roadmapId = roadmapId;
        this.phaseNumber = phaseNumber;
        this.status = status;
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public Integer getRoadmapId() {
        return roadmapId;
    }

    public void setRoadmapId(Integer roadmapId) {
        this.roadmapId = roadmapId;
    }

    public Integer getPhaseNumber() {
        return phaseNumber;
    }

    public void setPhaseNumber(Integer phaseNumber) {
        this.phaseNumber = phaseNumber;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }
}