---
marp: true
theme: default
paginate: true
header: "huynq03"
math: true

---

# Weekly Report

**Prepared by:** Huy Quang Nguyen    

**Date:** 10/10/2025

-  Paper : Avian-Inspired Grasping for Quadrotor Micro UAVs


---

# Last week

- Đọc bài báo và tìm hướng triển khai


---
# Task in progess

- Sử dụng dynamics và PID điều khiển quadrotor fly from A to B
- Xây dựng lại QP trong bài báo

---
# Điều khiển quadrotor fly from A to B

- Video demo

---
# Xây dựng lại QP

Bài toán QP đẳng thức:
$$
\min\; \tfrac12\,\mathbf c^\top \mathbf H\,\mathbf c + \mathbf f^\top \mathbf c
\quad\text{s.t.}\quad
\mathbf A_{\text{eq}}\mathbf c=\mathbf b_{\text{eq}}.
$$


- Đầu vào phụ thuộc **snap** $\big(y^{(4)}\big)$ 
- **Hàm mục tiêu** cho mỗi thành phần $i \in \{x_q,z_q,\beta\}$  
  $$
  J_i \;=\; \int_{t_0}^{t_f} \big(y_i^{(4)}(t)\big)^2\, dt.
  $$
- **Ràng buộc** :  
  *Start/Finish*: chốt vị trí, $\dot y=\ddot y=\dddot y=0$.  
  *Pickup* tại $t=t_{\text{pick}}$: $\beta=90^\circ$; $x_q,z_q$ chốt vị trí; **liên tục C³** qua nút.  
  *Trước pickup*: $\beta$ **chỉ vào mục tiêu** (look‑at).

---

## Tham số hoá đa thức theo đoạn 

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

## 

Với $r=4$: $\dfrac{d^r}{dt^r} t^k = k(k-1)(k-2)(k-3)\, t^{k-4}$ nếu $k\ge4$, ngược lại bằng $0$.  
Cho một đoạn dài $T$, phần tử Hessian (đối xứng) là
$$
Q_{ij} \;=\;
\begin{cases}
\dfrac{i!}{(i-4)!}\;\dfrac{j!}{(j-4)!}\;\dfrac{T^{\,i+j-7}}{\,i+j-7\,}, & i,j\ge4,\\[6pt]
0, & \text{khác.}
\end{cases}
$$


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