---
permalink: /home/
hidden: true
header:
  overlay_color: "#5e616c"
  overlay_image: /assets/images/background_1.jpg
excerpt: >
  Mon blog sur l'informatique, les sciences, l'intelligence artificielle, les technologies et bref tout ce que je peux trouver d'intéressant à raconter :fire: ... 
---

<h3 class="archive__subtitle">{{ site.data.ui-text[site.locale].recent_posts | default: "Recent Posts" }}</h3>

{% if paginator %}
  {% assign posts = paginator.posts %}
{% else %}
  {% assign posts = site.posts %}
{% endif %}

{% assign entries_layout = page.entries_layout | default: 'list' %}
<div class="entries-{{ entries_layout }}">
  {% for post in posts %}
    {% include archive-single.html type=entries_layout %}
  {% endfor %}
</div>

{% include paginator.html %}