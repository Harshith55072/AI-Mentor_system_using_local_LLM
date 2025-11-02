package com.ai.mentor.backend.controller;

import com.ai.mentor.backend.model.User;
import com.ai.mentor.backend.model.UserProgress;
import com.ai.mentor.backend.repository.UserProgressRepository;
import com.ai.mentor.backend.repository.UserRepository;
import com.ai.mentor.backend.service.JwtService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@CrossOrigin(origins = {"http://localhost:5500", "http://127.0.0.1:5500"}, allowCredentials = "true")
@RestController
@RequestMapping("/api/progress")
public class ProgressController {

    @Autowired
    private UserProgressRepository progressRepository;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private JwtService jwtService;

    /**
     * Get user's DSA progress
     */
    @GetMapping
    public ResponseEntity<?> getProgress(@RequestHeader("Authorization") String authHeader) {
        try {
            String token = authHeader.replace("Bearer ", "");
            String email = jwtService.extractUsername(token);

            User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found"));

            // Get or create progress
            UserProgress progress = progressRepository.findByUser(user)
                    .orElseGet(() -> {
                        UserProgress newProgress = new UserProgress(user);
                        return progressRepository.save(newProgress);
                    });

            // Return DTO
            ProgressDTO dto = new ProgressDTO(
                    progress.getDsaPart1Complete(),
                    progress.getDsaPart2Complete(),
                    progress.getDsaPart3Complete(),
                    progress.getDsaPart4Complete(),
                    progress.getProgressPercentage()
            );

            System.out.println("✅ Progress fetched for " + email + ": " + progress.getProgressPercentage() + "%");
            return ResponseEntity.ok(dto);

        } catch (Exception e) {
            System.err.println("❌ Error fetching progress: " + e.getMessage());
            return ResponseEntity.status(500).body("Error: " + e.getMessage());
        }
    }

    /**
     * Mark a DSA part as complete
     */
    @PostMapping("/complete/{partNumber}")
    public ResponseEntity<?> markComplete(
            @PathVariable int partNumber,
            @RequestHeader("Authorization") String authHeader) {

        try {
            String token = authHeader.replace("Bearer ", "");
            String email = jwtService.extractUsername(token);

            User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found"));

            // Get or create progress
            UserProgress progress = progressRepository.findByUser(user)
                    .orElseGet(() -> new UserProgress(user));

            // Mark the appropriate part as complete
            switch (partNumber) {
                case 1 -> progress.setDsaPart1Complete(true);
                case 2 -> progress.setDsaPart2Complete(true);
                case 3 -> progress.setDsaPart3Complete(true);
                case 4 -> progress.setDsaPart4Complete(true);
                default -> {
                    return ResponseEntity.badRequest().body("Invalid part number: " + partNumber);
                }
            }

            progressRepository.save(progress);

            System.out.println("✅ Marked DSA Part " + partNumber + " complete for " + email);
            System.out.println("📊 New progress: " + progress.getProgressPercentage() + "%");

            return ResponseEntity.ok(new ProgressDTO(
                    progress.getDsaPart1Complete(),
                    progress.getDsaPart2Complete(),
                    progress.getDsaPart3Complete(),
                    progress.getDsaPart4Complete(),
                    progress.getProgressPercentage()
            ));

        } catch (Exception e) {
            System.err.println("❌ Error marking complete: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(500).body("Error: " + e.getMessage());
        }
    }

    // DTO for progress response
    public static class ProgressDTO {
        private boolean part1Complete;
        private boolean part2Complete;
        private boolean part3Complete;
        private boolean part4Complete;
        private int progressPercentage;

        public ProgressDTO(boolean part1, boolean part2, boolean part3, boolean part4, int percentage) {
            this.part1Complete = part1;
            this.part2Complete = part2;
            this.part3Complete = part3;
            this.part4Complete = part4;
            this.progressPercentage = percentage;
        }

        // Getters
        public boolean isPart1Complete() { return part1Complete; }
        public boolean isPart2Complete() { return part2Complete; }
        public boolean isPart3Complete() { return part3Complete; }
        public boolean isPart4Complete() { return part4Complete; }
        public int getProgressPercentage() { return progressPercentage; }
    }
}