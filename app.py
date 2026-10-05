from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from svm_utils import (
    DATASET_LABELS,
    decision_grid,
    fit_and_evaluate,
    kernel_similarity,
    make_dataset,
    metrics_frame,
    poly_response,
)

st.set_page_config(
    page_title="BAT 3305 - Colazo | SVM Kernel Lab",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
    .hero {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        padding: 1.15rem 1.35rem;
        margin-bottom: .8rem;
        background: linear-gradient(135deg, rgba(90,90,90,.08), rgba(90,90,90,.02));
    }
    .hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: -.03em;}
    .hero p {margin: .25rem 0 0 0; opacity: .78; font-size: 1.05rem;}
    .concept-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: .85rem 1rem;
        height: 100%;
    }
    .concept-card strong {font-size: 1.02rem;}
    .small-note {opacity: .73; font-size: .9rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        padding: .75rem .85rem;
        border-radius: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>BAT 3305 - Colazo</h1>
  <p><strong>Support Vector Machines:</strong> Kernels, Margins & Decision Boundaries</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Experiment controls")
    dataset_name = st.selectbox(
        "Dataset",
        list(DATASET_LABELS.keys()),
        index=2,
        help="Each dataset emphasizes a different kind of decision boundary.",
    )
    st.caption(DATASET_LABELS[dataset_name])

    n_samples = st.slider("Observations", 80, 400, 180, 20)
    noise = st.slider("Noise", 0.01, 0.55, 0.16, 0.01)
    random_state = st.number_input("Random seed", 0, 9999, 42, 1)
    test_size = st.slider("Test-set share", 0.15, 0.40, 0.25, 0.05)

    st.divider()
    st.subheader("SVM settings")
    kernel_label = st.radio(
        "Kernel",
        ["Linear", "Polynomial", "RBF / Gaussian"],
        index=2,
    )
    kernel = {"Linear": "linear", "Polynomial": "poly", "RBF / Gaussian": "rbf"}[kernel_label]

    C = st.select_slider(
        "C — penalty for margin violations",
        options=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        value=3.0,
        help="Lower C favors a wider, softer margin. Higher C penalizes mistakes more strongly.",
    )

    gamma = 1.0
    degree = 3
    coef0 = 1.0
    if kernel in {"rbf", "poly"}:
        gamma = st.select_slider(
            "Gamma (γ) — locality / curvature",
            options=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            value=1.0,
            help="High γ makes each observation's influence more local; low γ makes it broader.",
        )
    if kernel == "poly":
        degree = st.slider("Polynomial degree", 2, 6, 3)
        coef0 = st.slider("coef0 — independent-term influence", 0.0, 3.0, 1.0, 0.25)

    st.divider()
    st.caption("Teaching tip: keep the dataset fixed and change one hyperparameter at a time.")

X, y = make_dataset(dataset_name, n_samples, noise, int(random_state))
results = fit_and_evaluate(
    X=X,
    y=y,
    kernel=kernel,
    C=C,
    gamma=gamma,
    degree=degree,
    coef0=coef0,
    test_size=test_size,
    random_state=int(random_state),
)


def draw_boundary():
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    xx, yy, grid = decision_grid(X, resolution=260)
    pred = results.pipeline.predict(grid).reshape(xx.shape)
    decision = results.pipeline.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], alpha=0.12)
    ax.contour(xx, yy, decision, levels=[0], linewidths=2.2)
    ax.contour(xx, yy, decision, levels=[-1, 1], linestyles="--", linewidths=1.1, alpha=0.75)

    ax.scatter(
        results.X_train[:, 0],
        results.X_train[:, 1],
        c=results.y_train,
        s=48,
        edgecolors="white",
        linewidths=0.55,
        label="Training observations",
    )
    ax.scatter(
        results.X_test[:, 0],
        results.X_test[:, 1],
        c=results.y_test,
        marker="X",
        s=80,
        linewidths=0.7,
        edgecolors="black",
        label="Test observations",
    )

    support = results.X_train[results.support_indices_train]
    ax.scatter(
        support[:, 0],
        support[:, 1],
        s=145,
        facecolors="none",
        edgecolors="black",
        linewidths=1.65,
        label="Support vectors",
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{kernel_label} SVM — decision boundary and margin")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.13)
    fig.tight_layout()
    return fig


def kernel_takeaway():
    if kernel == "linear":
        return (
            "The linear kernel can only produce a straight boundary in this two-feature space. "
            "It is a strong baseline, but it cannot bend around curved class structure."
        )
    if kernel == "poly":
        return (
            f"The degree-{degree} polynomial kernel creates feature interactions implicitly. "
            "Higher degree can add curvature quickly, so complexity can rise sharply."
        )
    return (
        "The RBF kernel measures local similarity. Gamma controls how quickly similarity fades with distance: "
        "small γ gives broad, smooth influence; large γ creates very local, flexible boundaries."
    )


def complexity_warning():
    if kernel == "rbf" and gamma >= 10 and C >= 10:
        return "⚠️ High C + high γ is a classic overfitting combination. Compare training fit with held-out test performance."
    if kernel == "poly" and degree >= 5 and C >= 10:
        return "⚠️ High-degree polynomial + high C can create a very rigid fit to the current sample."
    if C <= 0.1:
        return "ℹ️ Very low C deliberately tolerates more margin violations in exchange for a softer, wider-margin solution."
    return "✅ The current settings are a useful middle ground for exploring bias versus flexibility."

main_tab, intuition_tab, diagnostics_tab, lab_tab = st.tabs(
    ["Decision Boundary", "Kernel Intuition", "Diagnostics", "Student Lab"]
)

with main_tab:
    chart_col, expl_col = st.columns([1.7, 0.8], gap="large")
    with chart_col:
        st.pyplot(draw_boundary(), use_container_width=True)
        st.caption("Solid line = decision boundary. Dashed lines = SVM margin levels ±1. Ringed observations = support vectors. X markers = held-out test observations.")
    with expl_col:
        st.subheader("What changed?")
        st.write(kernel_takeaway())
        st.info(complexity_warning())
        st.markdown("**Current model**")
        st.write(f"Kernel: **{kernel_label}**")
        st.write(f"C: **{C:g}**")
        if kernel in {"rbf", "poly"}:
            st.write(f"Gamma: **{gamma:g}**")
        if kernel == "poly":
            st.write(f"Degree: **{degree}**")
            st.write(f"coef0: **{coef0:g}**")
        st.markdown("**Observe:**")
        st.markdown(
            "- Which points become support vectors?\n"
            "- Does the margin get wider or narrower?\n"
            "- Does the test accuracy move with training flexibility?\n"
            "- Is the extra boundary complexity actually useful?"
        )

    st.subheader("Performance snapshot")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Test accuracy", f"{results.metrics['Accuracy']:.1%}")
    m2.metric("Precision", f"{results.metrics['Precision']:.1%}")
    m3.metric("Recall", f"{results.metrics['Recall']:.1%}")
    m4.metric("F1", f"{results.metrics['F1']:.1%}")
    m5.metric(
        "Support vectors",
        f"{int(results.metrics['Support vectors'])}",
        help=f"{results.metrics['Support vector share']:.1%} of training observations",
    )

with intuition_tab:
    st.subheader("Why kernels matter")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="concept-card"><strong>Linear</strong><br><span class="small-note">Similarity is based on the ordinary dot product. The resulting separator is a straight line in 2D.</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="concept-card"><strong>Polynomial</strong><br><span class="small-note">Implicitly adds powers and feature interactions, letting the separating surface curve.</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="concept-card"><strong>RBF / Gaussian</strong><br><span class="small-note">Treats nearby points as more similar than distant points, enabling highly flexible local boundaries.</span></div>', unsafe_allow_html=True)

    st.markdown("### The kernel trick")
    st.write(
        "An SVM can behave as though it transformed the observations into a richer feature space without explicitly constructing every transformed feature. "
        "The kernel computes the needed similarities directly."
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### RBF: effect of γ")
        distances = np.linspace(0, 3, 180)
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for g in [0.1, 1.0, 10.0]:
            ax.plot(distances, kernel_similarity(distances, g), label=f"γ = {g:g}")
        ax.set_xlabel("Distance between observations")
        ax.set_ylabel("RBF similarity")
        ax.set_ylim(-0.02, 1.04)
        ax.grid(alpha=.15)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Large γ makes similarity collapse quickly with distance, so influence becomes highly local.")

    with right:
        st.markdown("#### Polynomial: effect of degree")
        dots = np.linspace(-1.0, 1.5, 220)
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for d in [2, 3, 5]:
            ax.plot(dots, poly_response(dots, gamma=1.0, degree=d, coef0=1.0), label=f"degree = {d}")
        ax.set_xlabel("Dot product between observations")
        ax.set_ylabel("Polynomial-kernel response")
        ax.grid(alpha=.15)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Higher degree amplifies nonlinear interaction structure much more aggressively.")

    st.markdown("### Three ideas students should leave with")
    st.markdown(
        "1. **C does not choose the kernel.** It controls how strongly margin violations are penalized.\n"
        "2. **Gamma matters only for kernels that use it**, especially RBF and polynomial.\n"
        "3. **More flexible is not automatically better.** Model selection should be judged on held-out performance, not merely on how neatly the training sample is separated."
    )

with diagnostics_tab:
    st.subheader("Held-out model diagnostics")
    left, right = st.columns([1, 1], gap="large")
    with left:
        cm = pd.DataFrame(
            results.confusion,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.markdown("#### Confusion matrix")
        st.dataframe(cm, use_container_width=True)
        st.caption("Rows are actual classes; columns are model predictions.")

    with right:
        st.markdown("#### Predictive metrics")
        metric_df = metrics_frame(results.metrics).copy()
        metric_df["Value"] = metric_df["Value"].map(lambda v: f"{v:.1%}")
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
        st.write(
            f"**{int(results.metrics['Support vectors'])}** of **{len(results.X_train)}** training observations "
            f"({results.metrics['Support vector share']:.1%}) are support vectors."
        )

    st.markdown("### How to read the support-vector count")
    st.write(
        "Support vectors are the training observations that actively determine the fitted boundary. "
        "A very large share can indicate that the classification problem is difficult, noisy, heavily overlapping, or that the chosen hyperparameters produce a complex boundary. "
        "The count is informative, but it is not by itself a measure of model quality."
    )

with lab_tab:
    st.subheader("Guided classroom experiments")
    st.write("Use the sidebar controls. Change **one variable at a time** and record what happens to the boundary, margin, support vectors, and test accuracy.")

    experiments = pd.DataFrame(
        [
            ["1. Can a straight line solve it?", "Concentric circles", "Linear", "C = 1", "Explain why changing C cannot rescue the fundamental shape mismatch."],
            ["2. Kernel rescue", "Concentric circles", "RBF", "C = 1; try γ = 0.1, 1, 10", "Identify underfitting, a useful middle setting, and possible overfitting."],
            ["3. Soft vs. hard penalty", "Linear separation", "Linear", "Compare C = 0.03 and C = 30", "Track margin width, mistakes, support-vector count, and test accuracy."],
            ["4. Polynomial flexibility", "XOR", "Polynomial", "Try degree 2, 3, 5", "Explain how degree changes the available boundary shapes."],
            ["5. Noise stress test", "Two moons", "RBF", "Increase noise gradually", "Ask when a more flexible boundary stops helping held-out performance."],
        ],
        columns=["Experiment", "Dataset", "Kernel", "Settings", "Question"],
    )
    st.dataframe(experiments, use_container_width=True, hide_index=True)

    st.markdown("### Exit questions")
    st.markdown(
        "1. What is a **support vector**, in practical terms?\n"
        "2. Why can a nonlinear kernel outperform a linear kernel without explicitly adding new columns to the dataset?\n"
        "3. How do **C** and **γ** play different roles?\n"
        "4. What visual signs suggest underfitting? What signs suggest overfitting?\n"
        "5. Why should the test observations not be used to choose a model repeatedly without caution?"
    )

st.divider()
st.caption("BAT 3305 - Colazo • Interactive SVM kernel demonstration • Synthetic data generated locally in the app")
