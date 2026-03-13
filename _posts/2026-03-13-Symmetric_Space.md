---
layout: post
title: "[ICLR 2026] Symmetric Space Learning for Combinatorial Generalization"
date: 2026-03-13
categories: research
---

[OpenReview](https://openreview.net/forum?id=e8t9F4vX9N&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))

[Project GitHub](https://github.com/LynAlpha/CartanFM)

<p align="center">
  <img src="{{ site.baseurl }}/assets/2026_ICLR_Jaehyoung_ver56.png" width="100%" alt="ICLR 2026 Poster" />
</p>

# TL;DR
Let's make ML models imagine unseen submanifolds through the lens of **Symmetric Spaces**!

## What is Combinatorial Generalization
Assume that you have seen every rose in the world and every blue flower in the world. Then, can you imagine the *blue rose*, which does not exist in the world? Human can imagine such "unobserved image" by recombining familiar features, however, machine learning models suffer from such Out-of-Distribution situation.

We define **Combinatorial Generalization (CG)** as an ability which is inferencing/reasoning/generating unseen sample with observed features. Such CG ability is important factor for achieving human-level intelligence, because it is directly related to foundation of human's creativity and imagination.

Existing approaches mainly rely on symmetry and group theoretic methods. Symmetry is powerful tool for generalizing rules learned from insufficient data to wild situation. Assume that we know the group element "making blue" and latent representation of "red rose". Then, we can obtain "blue rose" via acting the element on latent representation.

## Is Group Theory Enough?
However, we are missing principle part in these approaches. **Group action must be closed.** If our model learned the symmetry of given training data *perfectly*, then we cannot extend it into total dataset because the earned symmetry is just permutation over given training data. In our paper, we defined this problem as **Symmetry Generalization**. Key point is that we should focus on symmetry of total dataset not of given training dataset.

However, how can we anticipate the test distribution without direct observation? Our intuition is utilizing the **Manifold Hypothesis**. In most cases, we can assume that training dataset and test dataset lie on same data manifold. If we can observing exact data manifold, and explore unseen region of the manifold, we can access to the symmetry of total dataset.

## How about Explore Manifold with Group?
Then, the next step is how to we explore unseen region of the manifold. For this, we introduce **Homogeneous space**, the manifold equipped with transitive group action. That is, when a point on a homogeneous space is given then there always exists a group action which transform the point to any other point on manifold. The definition and characteristic of homogeneous space is perfectly aligns with our goal of symmetry generalization.

Moreover, we specify **Symmetric space** is the best choice for such exploration. Symmetric Space is a type of homogeneous space, which is the manifold equipped with geodesic symmetry. That is, when a point on a symmetric space is given then there always exist a geodesic between the point and any other point and *involution* (reflecting) over geodesic. We can extend learned symmetry on given data via geodesic symmetry, by reaching the opposite side of geodesic to unseen region.

## How Can We Force Symmetric Space?
Placeholder

## Don't Escape From Manifold!
Placeholder

## Is It Good?
Placeholder

## Then, What is the Next?
Placeholder