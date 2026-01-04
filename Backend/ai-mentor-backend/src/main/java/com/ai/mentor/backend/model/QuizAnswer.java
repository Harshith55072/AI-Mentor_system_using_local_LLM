package com.ai.mentor.backend.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "quiz_answers")
public class QuizAnswer {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "roadmap_id", nullable = false)
    private Integer roadmapId;

    @Column(name = "phase_number", nullable = false)
    private Integer phaseNumber;

    @Column(name = "score", nullable = false)
    private Integer score;

    @Column(name = "total_questions", nullable = false)
    private Integer totalQuestions;

    @Column(name = "answers", columnDefinition = "TEXT")
    private String answers; // JSON string of user answers

    @Column(name = "timestamp", nullable = false)
    private LocalDateTime timestamp;

    @Column(name = "passed", nullable = false)
    private Boolean passed;

    public QuizAnswer() {}

    public QuizAnswer(User user, Integer roadmapId, Integer phaseNumber,
                      Integer score, Integer totalQuestions, String answers, Boolean passed) {
        this.user = user;
        this.roadmapId = roadmapId;
        this.phaseNumber = phaseNumber;
        this.score = score;
        this.totalQuestions = totalQuestions;
        this.answers = answers;
        this.passed = passed;
        this.timestamp = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public User getUser() { return user; }
    public void setUser(User user) { this.user = user; }

    public Integer getRoadmapId() { return roadmapId; }
    public void setRoadmapId(Integer roadmapId) { this.roadmapId = roadmapId; }

    public Integer getPhaseNumber() { return phaseNumber; }
    public void setPhaseNumber(Integer phaseNumber) { this.phaseNumber = phaseNumber; }

    public Integer getScore() { return score; }
    public void setScore(Integer score) { this.score = score; }

    public Integer getTotalQuestions() { return totalQuestions; }
    public void setTotalQuestions(Integer totalQuestions) { this.totalQuestions = totalQuestions; }

    public String getAnswers() { return answers; }
    public void setAnswers(String answers) { this.answers = answers; }

    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }

    public Boolean getPassed() { return passed; }
    public void setPassed(Boolean passed) { this.passed = passed; }
}