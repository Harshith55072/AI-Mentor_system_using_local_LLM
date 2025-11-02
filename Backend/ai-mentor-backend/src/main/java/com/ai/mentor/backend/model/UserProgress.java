package com.ai.mentor.backend.model;

import jakarta.persistence.*;

@Entity
@Table(name = "user_progress")
public class UserProgress {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne
    @JoinColumn(name = "user_id", nullable = false, unique = true)
    private User user;

    @Column(name = "dsa_part1_complete")
    private Boolean dsaPart1Complete = false;

    @Column(name = "dsa_part2_complete")
    private Boolean dsaPart2Complete = false;

    @Column(name = "dsa_part3_complete")
    private Boolean dsaPart3Complete = false;

    @Column(name = "dsa_part4_complete")
    private Boolean dsaPart4Complete = false;

    public UserProgress() {}

    public UserProgress(User user) {
        this.user = user;
        this.dsaPart1Complete = false;
        this.dsaPart2Complete = false;
        this.dsaPart3Complete = false;
        this.dsaPart4Complete = false;
    }

    // Calculate progress percentage (0-100)
    public int getProgressPercentage() {
        int completed = 0;
        if (dsaPart1Complete) completed++;
        if (dsaPart2Complete) completed++;
        if (dsaPart3Complete) completed++;
        if (dsaPart4Complete) completed++;
        return (completed * 100) / 4; // 25% per part
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

    public Boolean getDsaPart1Complete() {
        return dsaPart1Complete;
    }

    public void setDsaPart1Complete(Boolean dsaPart1Complete) {
        this.dsaPart1Complete = dsaPart1Complete;
    }

    public Boolean getDsaPart2Complete() {
        return dsaPart2Complete;
    }

    public void setDsaPart2Complete(Boolean dsaPart2Complete) {
        this.dsaPart2Complete = dsaPart2Complete;
    }

    public Boolean getDsaPart3Complete() {
        return dsaPart3Complete;
    }

    public void setDsaPart3Complete(Boolean dsaPart3Complete) {
        this.dsaPart3Complete = dsaPart3Complete;
    }

    public Boolean getDsaPart4Complete() {
        return dsaPart4Complete;
    }

    public void setDsaPart4Complete(Boolean dsaPart4Complete) {
        this.dsaPart4Complete = dsaPart4Complete;
    }
}
