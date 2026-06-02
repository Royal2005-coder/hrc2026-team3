## Tổng quan

PR này tổng hợp code Task 1 vào nhánh staging để thầy Hải review trước khi merge vào main.

## Module owners

- Perception / JSON / Camera: @thanh-tai-435
- Planner / FSM / Bin mapping: @ckothuw123
- Planner Support / Transform / Gripper / IK: @tduybao7605
- Motion / Pick-place / Waypoints: @Bias209
- Evaluation / PM / Runner: @Royal2005-coder

## Reviewer chính

- @haintktl-HRC2026

## Nội dung cần review

- [ ] Perception JSON coordinate correctness
- [ ] Planner đọc JSON và map bin đúng
- [ ] Transform world/base frame đúng
- [ ] Right-arm motion mượt hơn
- [ ] IK / gripper execution ổn định
- [ ] Không conflict giữa module Perception - Planner - Motion
- [ ] Có thể chạy được pipeline Task 1 trên Isaac Sim

## Quy trình fix sau review

Nếu thầy comment lỗi ở module nào:

1. Người phụ trách module đó sửa trên nhánh cá nhân.
2. Người phụ trách push lại nhánh cá nhân.
3. PM merge lại nhánh đó vào staging.
4. PR staging → main tự cập nhật.
5. Thầy Hải review lại.
6. Khi OK, thầy Hải merge vào main.

## Ghi chú PM

Member không push trực tiếp main.  
Member không push trực tiếp staging.  
PM merge vào staging.  
Thầy Hải review staging và merge main.
