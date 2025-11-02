package com.ai.mentor.backend.model;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Message {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    private ChatSession chatSession;

    @Enumerated(EnumType.STRING)
    private Role role;  // USER or AI

    @Column(columnDefinition = "TEXT")
    private String content;

    private LocalDateTime timestamp;

    public enum Role {
        USER,
        AI
    }
}
