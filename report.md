# Report — Memory-FPS Agent

(작성 중 — 초안 노트. 최종 제출 전 정리 예정.)

## Limitations (draft notes)

- **Doorway bleed-through when VLM-captioning wall content.** When the agent
  stands close to and faces straight through a doorway, the adjacent room's
  walls/images/objects can appear just as large and sharp in the frame as
  content actually in the current room, so a naive "near/large = current
  room, far/small = other room" heuristic fails in that specific case
  (confirmed on real data: `seed0_step00216.png`, HUD says "Ebon Annex" but
  the frame is dominated by a different room's blue walls, two wall images,
  and a floor object, seen straight through the doorway). Our own pixel-based
  wall-color check (`agent/vision.py::sample_wall_color`) was not fooled
  (still measured the current room's true color by raw pixel count), but
  that doesn't stop the VLM from describing the clearly-visible content
  behind the doorway as if it belonged to the current room.
  Mitigations applied: (1) the VLM prompt explicitly instructs it to treat
  anything framed inside a rectangular doorway/portal opening as belonging
  to the other room, regardless of how near/sharp it looks; (2) the live
  agent only triggers a room's VLM capture after moving a bit away from the
  doorway it entered through, not right at the threshold. Neither is a full
  fix — a few frames captured while turning toward a doorway can still leak
  a neighboring room's content into the current room's record.
