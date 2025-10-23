---
marp: true
title: Minimum‑snap QP cho flat outputs (x_q, z_q, β)
description: KKT + ràng buộc “look‑at” và β ≥ 0
paginate: true
theme: default
math: katex
---

# **Constrained Optimization**
## Intuition behind the Lagrangian

**Source :** https://youtu.be/GR4ff0dTLTw?si=aE5MJU-05Lv37wba
**Date:** 16/10/2025

---

## **1. Bài toán cơ bản: Tối ưu không ràng buộc**

**Mục tiêu:** Tìm điểm cực tiểu (hoặc cực đại) của hàm mục tiêu $J(x)$.

**Phương pháp:**
- Tìm điểm mà tại đó **độ dốc (slope)** bằng 0.
- Đối với hàm nhiều biến, chúng ta tìm điểm mà **gradient** bằng vector không.
- $$\nabla J(x) = 0$$

**Trực quan:** Tìm điểm thấp nhất trong một thung lũng hoặc đỉnh của một ngọn đồi.



---

## **2. Thêm rào cản: Tối ưu có ràng buộc**

Bây giờ, chúng ta không thể tự do lựa chọn bất kỳ điểm nào. Các lựa chọn của chúng ta bị giới hạn bởi một hoặc nhiều điều kiện.

- **Hàm mục tiêu:** $J(x_1, x_2) = x_1^2 + x_2^2$ (Bề mặt hình cái bát)
- **Hàm ràng buộc:** $C(x_1, x_2) = x_1 + x_2 + 3 = 0$ (Một đường thẳng)

**Nhiệm vụ:** Tìm điểm thấp nhất trên "cái bát", nhưng chỉ được phép đi trên "đường thẳng" đã cho.



---

## **3. Giải pháp Trực quan: Điểm Tiếp tuyến**

Hãy tưởng tượng bạn đang đi bộ trên đường thẳng ràng buộc trên một bản đồ địa hình (contour map) của hàm mục tiêu.

- Điểm tối ưu (cực tiểu/cực đại) trên đường đi của bạn chính là nơi đường đi **tiếp tuyến** với một đường đồng mức (contour line).

**Tại sao?**
- Nếu đường đi của bạn cắt các đường đồng mức, nghĩa là bạn vẫn đang đi lên hoặc xuống dốc.
- Chỉ khi đường đi tiếp tuyến, tại khoảnh khắc đó, bạn không đi lên cũng không đi xuống.



---

## **4. Từ Trực quan đến Toán học: Gradient Song song**

Làm thế nào để biểu diễn "điểm tiếp tuyến" bằng toán học?

- **Gradient của hàm mục tiêu** ($\nabla J$) luôn **vuông góc** với đường đồng mức của nó.
- **Gradient của hàm ràng buộc** ($\nabla C$) luôn **vuông góc** với đường ràng buộc của nó.

Khi đường đi (ràng buộc) tiếp tuyến với đường đồng mức, hai vector gradient của chúng phải **song song** với nhau!

**Kết luận:** Tại điểm tối ưu, $\nabla J$ song song với $\nabla C$.

---

## **5. Nhân tử Lagrange ($\lambda$)**

Nếu hai vector là song song, thì vector này là một phiên bản co giãn (scaled version) của vector kia.

- Hằng số co giãn đó chính là **Nhân tử Lagrange**, ký hiệu là $\lambda$ (lambda).

Chúng ta có thể viết mối quan hệ song song dưới dạng phương trình:
$$\nabla J(x) = -\lambda \nabla C(x)$$

- $\lambda$ là "hệ số" giúp cân bằng độ lớn giữa hai vector gradient tại điểm tối ưu.

---

## **6. Hàm Lagrangian: Công cụ hợp nhất**

Từ phương trình trước, chúng ta có thể sắp xếp lại:
$$\nabla J(x) + \lambda \nabla C(x) = 0$$

Biểu thức này chính là gradient của một hàm mới, gọi là **hàm Lagrangian ($L$)**:
$$\nabla \big( J(x) + \lambda C(x) \big) = 0$$

**Hàm Lagrangian được định nghĩa là:**
$$L(x, \lambda) = J(x) + \lambda C(x)$$

=> Chúng ta đã chuyển bài toán **tối ưu có ràng buộc** thành bài toán tìm điểm dừng **không ràng buộc** của hàm $L$.

---

## **7. Ví dụ Thực tế**

1. **Thiết lập hàm Lagrangian:**
   $L(x_1, x_2, \lambda) = (x_1^2 + x_2^2) + \lambda(x_1 + x_2 + 3)$

2. **Tính đạo hàm riêng và cho bằng 0:**
   - $\frac{\partial L}{\partial x_1} = 2x_1 + \lambda = 0$
   - $\frac{\partial L}{\partial x_2} = 2x_2 + \lambda = 0$
   - $\frac{\partial L}{\partial \lambda} = x_1 + x_2 + 3 = 0$

3. **Giải hệ phương trình:**
   - $x_1 = -1.5$
   - $x_2 = -1.5$
   - $\lambda = 3$

Đây chính là điểm cực tiểu của bài toán có ràng buộc.

---

## **8. Tổng kết**

- **Bài toán có ràng buộc** yêu cầu tìm cực trị trên một đường/bề mặt giới hạn.
- **Ý tưởng cốt lõi:** Tại điểm tối ưu, đường ràng buộc sẽ **tiếp tuyến** với đường đồng mức của hàm mục tiêu.
- **Về mặt toán học:** Gradient của hàm mục tiêu ($\nabla J$) **song song** với gradient của hàm ràng buộc ($\nabla C$).
- **Nhân tử Lagrange ($\lambda$)** là hệ số tỷ lệ giữa hai gradient này.
- **Hàm Lagrangian ($L = J + \lambda C$)** là công cụ mạnh mẽ để biến bài toán có ràng buộc thành bài toán không ràng buộc, giúp chúng ta tìm ra lời giải.

