---
layout: page
title: blog
permalink: /blog/
description:
nav: true
nav_order: 3
---

<ul>
{% for post in site.posts %}
  <li>
    <a href="{{ post.url }}">{{ post.title }}</a>
    <span class="post-date">{{ post.date | date: "%b %d, %Y" }}</span>
  </li>
{% endfor %}
</ul>
