---
marp: true
title: Minimum‑snap QP cho flat outputs (x_q, z_q, β)
description: KKT + ràng buộc “look‑at” và β ≥ 0
paginate: true
theme: default
math: katex
---


# Minimum‑snap Quadratic Programming

---

## 1) Bài toán & giả thiết

- Hệ **phẳng** $\Rightarrow$ mọi trạng thái/điều khiển biểu diễn theo $y$ và đạo hàm của nó.  
- Đầu vào phụ thuộc **snap** $\big(y^{(4)}\big)$ $\Rightarrow$ quỹ đạo phải **trơn C³** (liên tục đến jerk).  
- **Hàm mục tiêu** cho mỗi thành phần $i \in \{x_q,z_q,\beta\}$  
  $$
  J_i \;=\; \int_{t_0}^{t_f} \big(y_i^{(4)}(t)\big)^2\, dt.
  $$
- **Ràng buộc** (như trong thí nghiệm):  
  *Start/Finish*: chốt vị trí, $\dot y=\ddot y=\dddot y=0$.  
  *Pickup* tại $t=t_{\text{pick}}$: $\beta=90^\circ$; $x_q,z_q$ chốt vị trí; **liên tục C³** qua nút.  
  *Trước pickup*: $\beta$ **chỉ vào mục tiêu** (look‑at).

---

## 2) Tham số hoá đa thức theo đoạn (bậc 7)

Chia thời gian thành 2 đoạn với $T_1=t_{\text{pick}}-t_0$, $T_2=t_f-t_{\text{pick}}$.  
Trên mỗi đoạn $s$, dùng đa thức bậc 7 theo thời gian cục bộ $\tau\in[0,T_s]$:
$$
p_s(\tau) \;=\; \sum_{k=0}^{7} a_{s,k}\,\tau^k.
$$

Gom hệ số của hai đoạn thành vector
$$
\mathbf c=\big[a_{1,0},\ldots,a_{1,7},\; a_{2,0},\ldots,a_{2,7}\big]^\top.
$$

---

## 3) Từ $\displaystyle \int \big\|p^{(4)}\big\|^2$ đến $\tfrac12\,\mathbf c^\top \mathbf Q\,\mathbf c$

Với $r=4$: $\dfrac{d^r}{dt^r} t^k = k(k-1)(k-2)(k-3)\, t^{k-4}$ nếu $k\ge4$, ngược lại bằng $0$.  
Cho một đoạn dài $T$, phần tử Hessian (đối xứng) là
$$
Q_{ij} \;=\;
\begin{cases}
\dfrac{i!}{(i-4)!}\;\dfrac{j!}{(j-4)!}\;\dfrac{T^{\,i+j-7}}{\,i+j-7\,}, & i,j\ge4,\\[6pt]
0, & \text{khác.}
\end{cases}
$$
$\Rightarrow$ $\mathbf Q$ là khối chéo theo các đoạn; chi phí mỗi 1D là $\tfrac12\,\mathbf c^\top\mathbf Q\,\mathbf c$.

---

## 4) Ràng buộc đẳng thức $\,\mathbf A_{\text{eq}}\mathbf c=\mathbf b_{\text{eq}}$

Hàng ràng buộc ứng với điều kiện $p^{(r)}(\tau^\*)=v^\*$ trên **đoạn $s$**:
$$
\underbrace{\big[\,\mathbf 0\;\cdots\;\mathbf 0,\; \boldsymbol\phi_r(\tau^\*)^\top,\; \mathbf 0\;\cdots\;\mathbf 0\,\big]}_{\text{khối thứ }s}
\mathbf c \;=\; v^\*,
$$
trong đó $\boldsymbol\phi_r(\tau) = \big[\tfrac{d^r}{d\tau^r}\tau^0,\ldots,\tfrac{d^r}{d\tau^r}\tau^7\big]$.

**Gói ràng buộc (mỗi 1D):**  
- **Đầu**: $p(0)=p_0$; $p^{(1)}(0)=p^{(2)}(0)=p^{(3)}(0)=0$.  
- **Pickup**: chốt $p_1(T_1)$; **C³ liên tục** $p_1^{(r)}(T_1)=p_2^{(r)}(0)$ với $r=0..3$.  
- **Cuối**: $p(T_1{+}T_2)=p_f$; $p^{(1)}=p^{(2)}=p^{(3)}=0$.  
- Với $\beta$: thêm **$\beta(t_{\text{pick}})=90^\circ$**.

---

## 5) Hệ KKT — GIẢI QP

Bài toán QP đẳng thức:
$$
\min\; \tfrac12\,\mathbf c^\top \mathbf H\,\mathbf c + \mathbf f^\top \mathbf c
\quad\text{s.t.}\quad
\mathbf A_{\text{eq}}\mathbf c=\mathbf b_{\text{eq}}.
$$

Hệ **KKT**:
$$
\begin{bmatrix}
\mathbf H & \mathbf A_{\text{eq}}^\top \\[2pt]
\mathbf A_{\text{eq}} & \mathbf 0
\end{bmatrix}
\begin{bmatrix}
\mathbf c\\ \boldsymbol\lambda
\end{bmatrix}
=
\begin{bmatrix}
-\mathbf f\\ \mathbf b_{\text{eq}}
\end{bmatrix},
$$
giải trực tiếp (ví dụ `numpy.linalg.solve`) thu $\mathbf c$.  
- Với $x_q,z_q$: $\mathbf f=\mathbf 0$.  
- Với $\beta$: có **ràng buộc mềm** $\Rightarrow \mathbf f\ne \mathbf 0$.

---

## 6) Ràng buộc mềm cho $\beta$ (fold vào $H,f$)

### 6.1 “Point‑at‑target” trước pickup
Mẫu $t_k<t_{\text{pick}}$:  
$$
\beta_{\text{LOS}}(t_k)=\operatorname{atan2}\big(z_T-z_q(t_k),\, x_T-x_q(t_k)\big).
$$
Thêm vào cost: $\sum_k \rho_{\text{pre}}\big(\mathbf r_k\mathbf c-\beta_{\text{LOS}}(t_k)\big)^2$.

### 6.2 Shaping sau pickup
Chọn mốc $t>t_{\text{pick}}$ để đỉnh $\approx125^\circ$ rồi về $90^\circ$.

---

### 6.3 Gộp LS vào $(\mathbf H,\mathbf f)$
Với $\mathbf R$ xếp các $\mathbf r_k$, $\mathbf y$ xếp mục tiêu, $\mathbf W=\operatorname{diag}(\sqrt{\rho_i})$:
$$
\boxed{\;\mathbf H \leftarrow \mathbf H + 2(\mathbf W\mathbf R)^\top(\mathbf W\mathbf R),\qquad
\mathbf f \leftarrow -2(\mathbf W\mathbf R)^\top(\mathbf W\mathbf y)\;}
$$
rồi **giải KKT** như trên.

---

## 7) Ép $\beta(t)\ge \beta_{\min}$ (không âm)

- **Cắt sàn** các mục tiêu LOS: $\beta_{\text{LOS}}\gets\max(\beta_{\text{LOS}},\,\beta_{\min})$.  
- **Hinge mềm (active‑set)**:
  1) Giải KKT $\Rightarrow \beta(t)$;  
  2) Tìm điểm lưới vi phạm $\beta(t)<\beta_{\min}$;  
  3) Thêm các mẫu với mục tiêu $\beta_{\min}$ và trọng số lớn $\rho_{\text{hinge}}$ vào $(\mathbf H,\mathbf f)$;  
  4) **Giải lại** KKT. Lặp đến khi không vi phạm.

---

## 8) Liên hệ với mã nguồn (qp2.py)

- `build_eq_qp_1d(...)` $\Rightarrow$ tạo $(\mathbf Q,\mathbf A_{\text{eq}},\mathbf b_{\text{eq}})$ cho từng 1D.  
- `solve_qp_equality(H,f,Aeq,beq)` $\Rightarrow$ **giải KKT**.  
- `add_soft_samples_to_objective(...)` $\Rightarrow$ gộp ràng buộc mềm vào $(H,f)$.  
- Quy trình $\beta$: **LOS (mềm)** + **shaping (mềm)** + **hinge β≥β_min** (lặp).

---

## 9) Tính $\theta^d(t)$ từ flatness (phụ lục)

Trung tâm khối hệ: $\displaystyle r_s=\frac{m_q r_q+m_g r_g}{m_s}$.  
Định hướng lực đẩy:
$$
\mathbf b_3=\frac{\ddot r_s+g\,\mathbf e_3}{\|\ddot r_s+g\,\mathbf e_3\|},\qquad
\theta=\operatorname{atan2}(b_{3x},\,b_{3z}).
$$
Vị trí gripper: $r_g=r_q+L_g[\cos\beta,\,0,\,-\sin\beta]^\top$.

---

## 10) Pseudocode

```
for y in {xq, zq, beta}:
  build Q, Aeq, beq
  if y in {xq, zq}: solve KKT (f=0)
  else:
    fold soft LOS & shaping into (H,f)
    repeat:
      solve KKT → beta(t)
      if any beta<beta_min: add hinge samples to (H,f)
      else: break
```

---

## 11) Gợi ý chọn tham số

- Bậc 7 đủ cho C³ + snap.  
- Dày mẫu LOS gần $t_{\text{pick}}$ $\Rightarrow$ sườn $\beta$ dốc.  
- Tinh chỉnh $\rho_{\text{pre}}$ (bám mục tiêu) và $\rho_{\text{post}}$ (đỉnh $\sim 125^\circ$).  
- Chọn $\beta_{\min}$ theo giới hạn cơ khí (thường $0^\circ$ hoặc $5^\circ$).