---
permalink: /projects/
layout: archive
classes: wide
---

<h3 class="archive__subtitle">{{ site.data.ui-text[site.locale].recent_projects | default: "Recent Projects" }}</h3>

{% if paginator %}
  {% assign projects = paginator.projects %}
{% else %}
  {% assign projects = site.projects %}
{% endif %}

{% assign entries_layout = page.entries_layout | default: 'list' %}
<div class="entries-{{ entries_layout }}">
  {% for post in projects %}
    {% include archive-single.html type=entries_layout %}
  {% endfor %}
</div>

{% include paginator.html %}