package com.ai.mentor.backend.controller;

import com.ai.mentor.backend.model.RoadmapProgress;
import com.ai.mentor.backend.model.User;
import com.ai.mentor.backend.repository.RoadmapProgressRepository;
import com.ai.mentor.backend.repository.UserRepository;
import com.ai.mentor.backend.service.JwtService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@CrossOrigin(origins = {"http://localhost:5500", "http://127.0.0.1:5500"}, allowCredentials = "true")
@RestController
@RequestMapping("/api/roadmap")
public class RoadmapController {

    @Autowired
    private RoadmapProgressRepository progressRepository;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private JwtService jwtService;

    /**
     * Test endpoint to verify authentication
     */
    @GetMapping("/test")
    public ResponseEntity<?> testAuth(@RequestHeader("Authorization") String authHeader) {
        try {
            String token = authHeader.replace("Bearer ", "");
            String email = jwtService.extractUsername(token);
            return ResponseEntity.ok("✅ Auth works! User: " + email);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("❌ Auth failed: " + e.getMessage());
        }
    }

    /**
     * Get all roadmap progress for the user
     */
    @GetMapping("/progress")
    public ResponseEntity<?> getProgress(@RequestHeader("Authorization") String authHeader) {
        try {
            String token = authHeader.replace("Bearer ", "");
            String email = jwtService.extractUsername(token);

            User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found"));

            List<RoadmapProgress> progressList = progressRepository.findByUser(user);

            // Convert to map for easy frontend access: {roadmapId: {phaseNumber: status}}
            Map<Integer, Map<Integer, String>> progressMap = new HashMap<>();

            for (RoadmapProgress progress : progressList) {
                progressMap
                        .computeIfAbsent(progress.getRoadmapId(), k -> new HashMap<>())
                        .put(progress.getPhaseNumber(), progress.getStatus());
            }

            System.out.println("✅ Loaded roadmap progress for: " + email);
            return ResponseEntity.ok(progressMap);

        } catch (Exception e) {
            System.err.println("❌ Error loading roadmap progress: " + e.getMessage());
            return ResponseEntity.status(500).body("Error: " + e.getMessage());
        }
    }

    /**
     * Mark a phase as "In Progress"
     */
    @PostMapping("/start/{roadmapId}/{phaseNumber}")
    public ResponseEntity<?> startPhase(
            @PathVariable Integer roadmapId,
            @PathVariable Integer phaseNumber,
            @RequestHeader("Authorization") String authHeader) {

        return updatePhaseStatus(roadmapId, phaseNumber, "IN_PROGRESS", authHeader);
    }

    /**
     * Mark a phase as "Completed"
     */
    @PostMapping("/complete/{roadmapId}/{phaseNumber}")
    public ResponseEntity<?> completePhase(
            @PathVariable Integer roadmapId,
            @PathVariable Integer phaseNumber,
            @RequestHeader("Authorization") String authHeader) {

        return updatePhaseStatus(roadmapId, phaseNumber, "COMPLETED", authHeader);
    }

    /**
     * Helper method to update phase status
     */
    private ResponseEntity<?> updatePhaseStatus(Integer roadmapId, Integer phaseNumber, String status, String authHeader) {
        try {
            String token = authHeader.replace("Bearer ", "");
            String email = jwtService.extractUsername(token);

            User user = userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found"));

            // Validate inputs
            if (roadmapId < 1 || roadmapId > 3) {
                return ResponseEntity.badRequest().body("Invalid roadmap ID: " + roadmapId);
            }
            if (phaseNumber < 1 || phaseNumber > 5) {
                return ResponseEntity.badRequest().body("Invalid phase number: " + phaseNumber);
            }

            // Find existing progress or create new
            RoadmapProgress progress = progressRepository
                    .findByUserAndRoadmapIdAndPhaseNumber(user, roadmapId, phaseNumber)
                    .orElse(new RoadmapProgress(user, roadmapId, phaseNumber, "NOT_STARTED"));

            progress.setStatus(status);
            progressRepository.save(progress);

            System.out.println("✅ Updated Roadmap " + roadmapId + ", Phase " + phaseNumber +
                    " to " + status + " for " + email);

            return ResponseEntity.ok(Map.of(
                    "roadmapId", roadmapId,
                    "phaseNumber", phaseNumber,
                    "status", status
            ));

        } catch (Exception e) {
            System.err.println("❌ Error updating phase: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(500).body("Error: " + e.getMessage());
        }
    }
}