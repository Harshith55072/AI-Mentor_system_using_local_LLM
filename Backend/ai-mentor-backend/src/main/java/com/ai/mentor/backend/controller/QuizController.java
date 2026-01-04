package com.ai.mentor.backend.controller;

import com.ai.mentor.backend.model.QuizAnswer;
import com.ai.mentor.backend.model.User;
import com.ai.mentor.backend.repository.QuizAnswerRepository;
import com.ai.mentor.backend.repository.UserRepository;
import com.ai.mentor.backend.service.JwtService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.Optional;

@CrossOrigin(origins = {"http://localhost:5500", "http://127.0.0.1:5500"}, allowCredentials = "true")
@RestController
@RequestMapping("/api/quiz")
public class QuizController {

    @Autowired
    private QuizAnswerRepository quizAnswerRepository;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private JwtService jwtService;

    /**
     * Submit quiz answers
     */
    @PostMapping("/submit")
    public ResponseEntity<?> submitQuiz(
            @RequestBody QuizSubmitRequest request,
            @RequestHeader("Authorization") String authHeader) {

        try {
            String token = authHeader.replace("Bearer ", "");
            String email = jwtService.extractUsername(token);

            User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found"));

            // Check if already submitted
            Optional<QuizAnswer> existing = quizAnswerRepository
                    .findByUserAndRoadmapIdAndPhaseNumber(user, request.getRoadmapId(), request.getPhaseNumber());

            QuizAnswer quizAnswer;
            if (existing.isPresent()) {
                // Update existing
                quizAnswer = existing.get();
                quizAnswer.setScore(request.getScore());
                quizAnswer.setTotalQuestions(request.getTotalQuestions());
                quizAnswer.setAnswers(request.getAnswers());
                quizAnswer.setPassed(request.getPassed());
            } else {
                // Create new
                quizAnswer = new QuizAnswer(
                        user,
                        request.getRoadmapId(),
                        request.getPhaseNumber(),
                        request.getScore(),
                        request.getTotalQuestions(),
                        request.getAnswers(),
                        request.getPassed()
                );
            }

            quizAnswerRepository.save(quizAnswer);

            System.out.println("✅ Quiz submitted: Roadmap " + request.getRoadmapId() +
                    ", Phase " + request.getPhaseNumber() +
                    ", Score: " + request.getScore() + "/" + request.getTotalQuestions());

            return ResponseEntity.ok(Map.of(
                    "success", true,
                    "score", request.getScore(),
                    "passed", request.getPassed()
            ));

        } catch (Exception e) {
            System.err.println("❌ Error submitting quiz: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * Get quiz result for specific phase
     */
    @GetMapping("/result/{roadmapId}/{phaseNumber}")
    public ResponseEntity<?> getQuizResult(
            @PathVariable Integer roadmapId,
            @PathVariable Integer phaseNumber,
            @RequestHeader("Authorization") String authHeader) {

        try {
            String token = authHeader.replace("Bearer ", "");
            String email = jwtService.extractUsername(token);

            User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found"));

            Optional<QuizAnswer> quizAnswer = quizAnswerRepository
                    .findByUserAndRoadmapIdAndPhaseNumber(user, roadmapId, phaseNumber);

            if (quizAnswer.isPresent()) {
                QuizAnswer qa = quizAnswer.get();
                return ResponseEntity.ok(Map.of(
                        "exists", true,
                        "score", qa.getScore(),
                        "totalQuestions", qa.getTotalQuestions(),
                        "passed", qa.getPassed(),
                        "timestamp", qa.getTimestamp().toString()
                ));
            } else {
                return ResponseEntity.ok(Map.of("exists", false));
            }

        } catch (Exception e) {
            System.err.println("❌ Error fetching quiz result: " + e.getMessage());
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * Get all quiz results for user
     */
    @GetMapping("/results")
    public ResponseEntity<?> getAllQuizResults(@RequestHeader("Authorization") String authHeader) {
        try {
            String token = authHeader.replace("Bearer ", "");
            String email = jwtService.extractUsername(token);

            User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found"));

            List<QuizAnswer> quizAnswers = quizAnswerRepository.findByUser(user);

            return ResponseEntity.ok(quizAnswers.stream().map(qa -> Map.of(
                    "roadmapId", qa.getRoadmapId(),
                    "phaseNumber", qa.getPhaseNumber(),
                    "score", qa.getScore(),
                    "totalQuestions", qa.getTotalQuestions(),
                    "passed", qa.getPassed()
            )).toList());

        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    // Request DTO
    public static class QuizSubmitRequest {
        private Integer roadmapId;
        private Integer phaseNumber;
        private Integer score;
        private Integer totalQuestions;
        private String answers;
        private Boolean passed;

        // Getters and Setters
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

        public Boolean getPassed() { return passed; }
        public void setPassed(Boolean passed) { this.passed = passed; }
    }
}