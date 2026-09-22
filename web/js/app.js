/**
 * Faculty-Course Dual-Layer Recommender (M4-v2)
 * Client-side Controller & UI State Engine
 */

class RecommenderApp {
  constructor() {
    this.allModules = [];
    this.allCourses = [];
    this.filteredModules = [];
    this.selectedModule = null;
    this.summaryData = null;

    this.currentPage = 1;
    this.pageSize = 25;

    // DOM Elements
    this.elements = {
      moduleList: document.getElementById('module-list'),
      detailPanel: document.getElementById('detail-panel'),
      emptyState: document.getElementById('empty-state'),
      detailContent: document.getElementById('detail-content'),
      resultsCount: document.getElementById('results-count'),
      pageIndicator: document.getElementById('page-indicator'),
      prevPageBtn: document.getElementById('prev-page-btn'),
      nextPageBtn: document.getElementById('next-page-btn'),
      searchInput: document.getElementById('module-search'),
      clearSearchBtn: document.getElementById('clear-search-btn'),
      departmentFilter: document.getElementById('department-filter'),
      courseFilter: document.getElementById('course-filter'),
      aspectBarContainer: document.getElementById('aspect-bar-container'),
      metricsPanel: document.getElementById('metrics-panel'),
      toggleStatsBtn: document.getElementById('toggle-stats-btn'),
      themeToggle: document.getElementById('theme-toggle'),
    };

    this.init();
  }

  async init() {
    this.setupEventListeners();
    await this.loadData();
    this.populateDepartmentFilter();
    this.populateCourseFilter('ALL');
    this.renderAspectMeters();
    this.applyFilters();

    // Default select first module or CS5707 if present
    const defaultMod = this.allModules.find(m => m.module_code === 'CS5707') || this.allModules[0];
    if (defaultMod) {
      this.selectModule(defaultMod);
    }
  }

  setupEventListeners() {
    // Search input debounce
    let searchDebounce;
    this.elements.searchInput.addEventListener('input', (e) => {
      clearTimeout(searchDebounce);
      const val = e.target.value.trim();
      this.elements.clearSearchBtn.classList.toggle('visible', val.length > 0);
      searchDebounce = setTimeout(() => {
        this.currentPage = 1;
        this.applyFilters();
      }, 200);
    });

    // Clear search
    this.elements.clearSearchBtn.addEventListener('click', () => {
      this.elements.searchInput.value = '';
      this.elements.clearSearchBtn.classList.remove('visible');
      this.currentPage = 1;
      this.applyFilters();
    });

    // Department Filter Change -> Dynamically updates Courses dropdown!
    this.elements.departmentFilter.addEventListener('change', () => {
      const selectedDept = this.elements.departmentFilter.value;
      this.populateCourseFilter(selectedDept);
      this.currentPage = 1;
      this.applyFilters();
    });

    // Course Filter Change
    this.elements.courseFilter.addEventListener('change', () => {
      this.currentPage = 1;
      this.applyFilters();
    });

    // Pagination
    this.elements.prevPageBtn.addEventListener('click', () => {
      if (this.currentPage > 1) {
        this.currentPage--;
        this.renderModuleList();
      }
    });

    this.elements.nextPageBtn.addEventListener('click', () => {
      const maxPage = Math.ceil(this.filteredModules.length / this.pageSize);
      if (this.currentPage < maxPage) {
        this.currentPage++;
        this.renderModuleList();
      }
    });

    // Toggle Metrics Panel
    this.elements.toggleStatsBtn.addEventListener('click', () => {
      this.elements.metricsPanel.classList.toggle('open');
      const isOpen = this.elements.metricsPanel.classList.contains('open');
      this.elements.toggleStatsBtn.innerHTML = isOpen 
        ? '<span class="icon">✕</span> Close Metrics' 
        : '<span class="icon">📊</span> University Metrics';
    });

    // Theme Switcher
    this.elements.themeToggle.addEventListener('click', () => {
      document.body.classList.toggle('light-theme');
      document.body.classList.toggle('dark-theme');
      const isLight = document.body.classList.contains('light-theme');
      this.elements.themeToggle.querySelector('.theme-icon').textContent = isLight ? '☀️' : '🌙';
    });
  }

  async loadData() {
    try {
      const cacheBust = Date.now();
      const [recsRes, summaryRes, coursesRes] = await Promise.all([
        fetch(`data/recommendations.json?t=${cacheBust}`),
        fetch(`data/summary.json?t=${cacheBust}`),
        fetch(`data/courses.json?t=${cacheBust}`),
      ]);

      if (recsRes.ok) {
        this.allModules = await recsRes.json();
      }

      if (summaryRes.ok) {
        this.summaryData = await summaryRes.json();
      }

      if (coursesRes.ok) {
        this.allCourses = await coursesRes.json();
      }
    } catch (err) {
      console.error('Failed to load dataset:', err);
    }
  }

  populateDepartmentFilter() {
    const departments = new Set();
    this.allModules.forEach(m => {
      (m.module_departments || []).forEach(d => {
        if (d && d.trim()) departments.add(d.trim());
      });
    });

    const sortedDepts = Array.from(departments).sort();
    sortedDepts.forEach(dept => {
      const opt = document.createElement('option');
      opt.value = dept;
      opt.textContent = dept;
      this.elements.departmentFilter.appendChild(opt);
    });
  }

  populateCourseFilter(selectedDepartment) {
    this.elements.courseFilter.innerHTML = '';

    let availableCourses = [];
    if (selectedDepartment === 'ALL') {
      availableCourses = this.allCourses;
      const allOpt = document.createElement('option');
      allOpt.value = 'ALL';
      allOpt.textContent = `All Courses (${availableCourses.length})`;
      this.elements.courseFilter.appendChild(allOpt);
    } else {
      // Find courses linked directly or via module offerings in that department
      const courseIdSet = new Set();

      // 1. Direct department match
      this.allCourses.forEach(c => {
        if ((c.departments || []).includes(selectedDepartment)) {
          courseIdSet.add(c.course_id);
        }
      });

      // 2. Module-linked courses in that department
      this.allModules.forEach(m => {
        if ((m.module_departments || []).includes(selectedDepartment)) {
          (m.courses || []).forEach(c => {
            if (c.course_id) courseIdSet.add(c.course_id);
          });
        }
      });

      availableCourses = this.allCourses.filter(c => courseIdSet.has(c.course_id));

      const allOpt = document.createElement('option');
      allOpt.value = 'ALL';
      allOpt.textContent = `All Courses in ${selectedDepartment} (${availableCourses.length})`;
      this.elements.courseFilter.appendChild(allOpt);
    }

    availableCourses.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.course_id;
      opt.textContent = c.course_title;
      this.elements.courseFilter.appendChild(opt);
    });

    this.elements.courseFilter.value = 'ALL';
  }

  renderAspectMeters() {
    if (!this.summaryData || !this.summaryData.aspect_distribution) return;

    const dist = this.summaryData.aspect_distribution;
    const maxCount = Math.max(...Object.values(dist));
    this.elements.aspectBarContainer.innerHTML = '';

    Object.entries(dist).forEach(([aspect, count]) => {
      const pct = Math.round((count / maxCount) * 100);
      const item = document.createElement('div');
      item.className = 'aspect-bar-item';
      item.innerHTML = `
        <div class="aspect-bar-info">
          <span class="aspect-bar-title">${aspect}</span>
          <span class="aspect-bar-count">${count} pairings</span>
        </div>
        <div class="aspect-bar-track">
          <div class="aspect-bar-fill" style="width: ${pct}%"></div>
        </div>
      `;
      this.elements.aspectBarContainer.appendChild(item);
    });
  }

  applyFilters() {
    const query = this.elements.searchInput.value.trim().toLowerCase();
    const selectedDept = this.elements.departmentFilter.value;
    const selectedCourseId = this.elements.courseFilter.value;

    this.filteredModules = this.allModules.filter(m => {
      const code = (m.module_code || '').toLowerCase();
      const title = (m.module_title || '').toLowerCase();
      const id = (m.module_id || '').toLowerCase();

      // Search matching
      const matchesSearch = !query || code === query || code.includes(query) || title.includes(query) || id.includes(query);
      if (!matchesSearch) return false;

      // Department matching
      if (selectedDept !== 'ALL') {
        const depts = m.module_departments || [];
        if (!depts.includes(selectedDept)) return false;
      }

      // Course matching
      if (selectedCourseId !== 'ALL') {
        const modCourses = m.courses || [];
        const matchesCourse = modCourses.some(c => c.course_id === selectedCourseId);
        if (!matchesCourse) return false;
      }

      return true;
    });

    this.renderModuleList();

    // Auto-select first matching module if currently selected module is no longer in filtered list
    if (this.filteredModules.length > 0) {
      const stillVisible = this.selectedModule && this.filteredModules.some(m => m.module_id === this.selectedModule.module_id);
      if (!stillVisible) {
        this.selectModule(this.filteredModules[0]);
      }
    } else {
      this.selectedModule = null;
      if (this.elements.emptyState && this.elements.detailContent) {
        this.elements.emptyState.style.display = 'flex';
        this.elements.detailContent.style.display = 'none';
      }
    }
  }

  renderModuleList() {
    const total = this.filteredModules.length;
    const maxPage = Math.max(1, Math.ceil(total / this.pageSize));
    if (this.currentPage > maxPage) this.currentPage = maxPage;

    const startIdx = (this.currentPage - 1) * this.pageSize;
    const endIdx = Math.min(startIdx + this.pageSize, total);
    const pagedItems = this.filteredModules.slice(startIdx, endIdx);

    this.elements.resultsCount.textContent = `${total} module${total === 1 ? '' : 's'} found`;
    this.elements.pageIndicator.textContent = `Page ${this.currentPage} of ${maxPage}`;
    this.elements.prevPageBtn.disabled = this.currentPage <= 1;
    this.elements.nextPageBtn.disabled = this.currentPage >= maxPage;

    this.elements.moduleList.innerHTML = '';

    if (pagedItems.length === 0) {
      this.elements.moduleList.innerHTML = `
        <div style="padding: 2rem 1rem; text-align: center; color: var(--text-muted); font-size: 0.85rem;">
          No modules match your current search and filter criteria.
        </div>
      `;
      return;
    }

    pagedItems.forEach(m => {
      const isSelected = this.selectedModule && this.selectedModule.module_id === m.module_id;
      const collabs = (m.layer_2_intelligent_collaboration || {}).cross_department_collaborations || [];
      const isExempt = !m.collaboration_eligible;
      const lead = (m.layer_1_primary_delivery || {}).lead_staff_name || 'Unassigned';

      let statusBadge = '';
      if (isExempt) {
        statusBadge = '<span class="badge badge-gray badge-xs">Exempt</span>';
      } else if (collabs.length > 0) {
        statusBadge = `<span class="badge badge-layer2 badge-xs">${collabs.length} Guest${collabs.length === 1 ? '' : 's'}</span>`;
      } else {
        statusBadge = '<span class="badge badge-warn badge-xs">Internal Lead Only</span>';
      }

      const card = document.createElement('div');
      card.className = `module-item-card ${isSelected ? 'active' : ''}`;
      card.innerHTML = `
        <div class="item-top-row">
          <span class="item-code">[${m.module_code || 'N/A'}]</span>
          ${statusBadge}
        </div>
        <div class="item-title">${m.module_title}</div>
        <div class="item-dept">${(m.module_departments || [])[0] || 'Unknown Department'}</div>
        <div class="item-footer-row">
          <span class="item-lead">Lead: ${lead}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        this.selectModule(m);
      });

      this.elements.moduleList.appendChild(card);
    });
  }

  selectModule(moduleData) {
    this.selectedModule = moduleData;

    // Update active highlight in list
    document.querySelectorAll('.module-item-card').forEach(c => c.classList.remove('active'));
    const activeEl = Array.from(this.elements.moduleList.children).find(c => 
      c.querySelector('.item-code')?.textContent === `[${moduleData.module_code || 'N/A'}]`
    );
    if (activeEl) activeEl.classList.add('active');

    this.renderModuleDetail(moduleData);
  }

  renderModuleDetail(m) {
    this.elements.emptyState.style.display = 'none';
    this.elements.detailContent.style.display = 'block';

    const l1 = m.layer_1_primary_delivery || {};
    const l2 = m.layer_2_intelligent_collaboration || {};
    const leadScore = l1.top_5_internal_recommendations?.[0]?.m3_kg_score || 0.0;
    const isExempt = !m.collaboration_eligible;
    const gaps = l2.curriculum_gaps_identified || [];
    const collabs = l2.cross_department_collaborations || [];
    const modCourses = m.courses || [];

    // Format linked degree programmes
    let coursesHtml = '';
    if (modCourses.length > 0) {
      coursesHtml = `
        <div class="banner-courses-block" style="margin-top: 0.85rem; padding-top: 0.75rem; border-top: 1px dashed var(--border-color);">
          <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.35rem;">Associated Degree Programmes (${modCourses.length})</div>
          <div style="display: flex; flex-wrap: wrap; gap: 0.35rem;">
            ${modCourses.map(c => `
              <span class="badge badge-accent badge-xs">
                ${c.course_title} ${c.year_of_study ? `(Year ${c.year_of_study})` : ''} ${c.module_type ? `&bull; ${c.module_type}` : ''}
              </span>
            `).join('')}
          </div>
        </div>
      `;
    }

    // Format top 5 internal faculty rows
    let top5Html = '';
    (l1.top_5_internal_recommendations || []).forEach(r => {
      const sharedTopics = (r.shared_topics || []).map(t => `<span class="topic-tag">${t}</span>`).join('');
      const explanationText = this.generateInternalFacultyExplanation(
        r,
        r.rank,
        m.module_title,
        (m.module_departments || [])[0] || 'Department'
      );
      const rProfileUrl = r.profile_url || (r.staff_id ? `https://www.brunel.ac.uk/people/${r.staff_id}` : '#');

      top5Html += `
        <div class="internal-faculty-row">
          <span class="rank-badge">#${r.rank}</span>
          <div class="internal-faculty-info">
            <div class="internal-faculty-header-line">
              <div class="faculty-name-row">
                <strong>${r.full_name}</strong> 
                <span class="faculty-pos-dept">(${r.position || 'Academic Staff'} &bull; ${r.department_name})</span>
                <a href="${rProfileUrl}" target="_blank" rel="noopener noreferrer" class="profile-link-badge" title="View ${r.full_name}'s Brunel Profile">
                  Profile ↗
                </a>
              </div>
              <span class="aspect-bar-count">Score: ${(r.m3_kg_score || 0).toFixed(4)}</span>
            </div>
            ${sharedTopics ? `<div class="topic-tags">${sharedTopics}</div>` : ''}
            <div class="internal-faculty-rationale">${explanationText}</div>
          </div>
        </div>
      `;
    });

    // Format gaps
    let gapsHtml = '';
    if (gaps.length > 0) {
      gapsHtml = `
        <div class="curriculum-gaps-block">
          <div class="ranking-title">Identified Curriculum Gaps &amp; Opportunities</div>
          <div class="gaps-list">
            ${gaps.map(g => `
              <div class="gap-pill">
                <span class="gap-title">${g.aspect_title}</span>
                <span class="badge ${g.need_level.includes('High') ? 'badge-layer2' : 'badge-accent'} badge-xs">${g.need_level}</span>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }

    // Format Collaborators or Exemption
    let layer2BodyHtml = '';
    if (isExempt) {
      layer2BodyHtml = `
        <div class="exemption-banner">
          <strong>Pedagogical Exemption Status</strong>
          ${l2.rationale || 'Independent student research project / placement unit; requires 1-on-1 supervision.'}
        </div>
      `;
    } else if (collabs.length > 0) {
      layer2BodyHtml = `
        ${gapsHtml}
        <div class="ranking-title" style="margin-top: 1rem;">Recommended Cross-Department Collaborators (${collabs.length})</div>
        <div class="collab-cards-list">
          ${collabs.map(c => {
            const dynamicRationale = this.generateCollabRationale(
              c,
              l1.lead_staff_name || 'Primary Lead',
              (m.module_departments || [])[0] || 'Department',
              m.module_title
            );
            const cProfileUrl = c.profile_url || (c.staff_id ? `https://www.brunel.ac.uk/people/${c.staff_id}` : '#');
            return `
              <div class="collab-card">
                <div class="collab-card-header">
                  <div class="collab-name-dept">
                    <div class="faculty-name-row">
                      <h4>${c.full_name}</h4>
                      <a href="${cProfileUrl}" target="_blank" rel="noopener noreferrer" class="profile-link-badge profile-link-collab" title="View ${c.full_name}'s Brunel Profile">
                        Profile ↗
                      </a>
                    </div>
                    <div class="collab-dept">${c.position || 'Faculty Member'} &bull; ${c.department_name}</div>
                  </div>
                  <div class="collab-aspect-badge">${c.aspect_title}</div>
                </div>
                <div class="collab-rationale">${dynamicRationale}</div>
                <div class="collab-scores-row">
                  <div class="collab-score-item">Aspect Affinity: <strong>${(c.aspect_affinity_score || 0).toFixed(4)}</strong></div>
                  <div class="collab-score-item">Collaboration Suitability: <strong>${(c.collaboration_suitability_score || 0).toFixed(4)}</strong></div>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      `;
    } else {
      layer2BodyHtml = `
        <div class="exemption-banner">
          <strong>No Cross-Department Gap Identified</strong>
          The primary lead's domain knowledge and departmental expertise fully saturate the curriculum requirements.
        </div>
      `;
    }

    const leadProfileUrl = l1.top_5_internal_recommendations?.[0]?.profile_url || (l1.lead_staff_id ? `https://www.brunel.ac.uk/people/${l1.lead_staff_id}` : '#');

    this.elements.detailContent.innerHTML = `
      <!-- Header Banner -->
      <div class="detail-banner">
        <div class="banner-meta-row">
          <span class="banner-code-badge">[${m.module_code || 'N/A'}]</span>
          <span class="badge ${isExempt ? 'badge-gray' : 'badge-layer1'}">${m.module_classification.replace(/_/g, ' ').toUpperCase()}</span>
        </div>
        <h2 class="banner-title">${m.module_title}</h2>
        <div class="banner-info-grid">
          <div class="banner-info-item">Department: <strong>${(m.module_departments || [])[0] || 'Unknown'}</strong></div>
          <div class="banner-info-item">College: <strong>${(m.module_colleges || [])[0] || 'Unknown'}</strong></div>
          <div class="banner-info-item">Module ID: <strong>${m.module_id}</strong></div>
        </div>
        ${coursesHtml}
      </div>

      <!-- Dual Layer Grid -->
      <div class="layers-container">
        <!-- Layer 1: Primary Delivery -->
        <div class="layer-card layer-1-card">
          <div class="layer-header">
            <div class="layer-title-wrap">
              <span class="badge badge-layer1">LAYER 1</span>
              <h3>Primary Curriculum Allocation (Departmental Fit)</h3>
            </div>
            <span class="badge badge-layer1">Sole / Lead Allocation</span>
          </div>
          <div class="layer-body">
            <div class="lead-faculty-card">
              <div class="faculty-name-dept">
                <div class="faculty-name-row">
                  <h4>${l1.lead_staff_name}</h4>
                  <a href="${leadProfileUrl}" target="_blank" rel="noopener noreferrer" class="profile-link-badge" title="View ${l1.lead_staff_name}'s Brunel Profile">
                    Brunel Profile ↗
                  </a>
                </div>
                <div class="faculty-dept-pos">${l1.lead_department} &bull; ${l1.top_5_internal_recommendations?.[0]?.position || 'Academic Staff'}</div>
              </div>
              <div class="faculty-score-pill">
                <div class="score-num">${leadScore.toFixed(4)}</div>
                <div class="score-label">M3-KG Score</div>
              </div>
            </div>

            <div class="top5-ranking-section">
              <div class="ranking-title">Top-5 Departmental Faculty Ranking</div>
              <div class="internal-faculty-list">
                ${top5Html}
              </div>
            </div>
          </div>
        </div>

        <!-- Layer 2: Intelligent Collaboration -->
        <div class="layer-card layer-2-card">
          <div class="layer-header">
            <div class="layer-title-wrap">
              <span class="badge badge-layer2">LAYER 2</span>
              <h3>Intelligent Cross-Department Collaboration</h3>
            </div>
            <span class="badge ${isExempt ? 'badge-gray' : (collabs.length > 0 ? 'badge-layer2' : 'badge-warn')}">
              ${l2.status || 'Active Co-Delivery'}
            </span>
          </div>
          <div class="layer-body">
            ${layer2BodyHtml}
          </div>
        </div>
      </div>
    `;
  }

  generateInternalFacultyExplanation(faculty, rank, moduleTitle, deptName) {
    const topics = faculty.shared_topics || [];
    const score = faculty.m3_kg_score || 0.0;
    const topicsStr = topics.length > 0 ? topics.slice(0, 3).map(t => `'${t}'`).join(', ') : 'core syllabus topics';

    if (rank === 1) {
      return `<strong>Primary Module Lead:</strong> Highest ${deptName} Knowledge Graph alignment (score: ${score.toFixed(3)}) with strong syllabus alignment in ${topicsStr}. Primary candidate for module leadership, lecture delivery, and syllabus management.`;
    } else if (rank === 2) {
      return `<strong>Core Co-Lecturer:</strong> Close departmental alignment in ${topicsStr} (score: ${score.toFixed(3)}). Ideal partner for co-teaching theoretical units and leading problem-solving tutorial tracks.`;
    } else if (rank === 3) {
      return `<strong>Lab & Seminar Lead:</strong> High domain alignment with syllabus topics (${topicsStr}). Well-equipped to supervise hands-on laboratory sessions and continuous coursework assessments.`;
    } else {
      return `<strong>Specialist Seminar Support:</strong> Complementary departmental profile in ${topicsStr} (score: ${score.toFixed(3)}). Available as alternate lecturer or specialist coursework marker.`;
    }
  }

  generateCollabRationale(collab, leadName, leadDept, moduleTitle) {
    const pos = collab.position || 'Faculty Member';
    const dept = collab.department_name || 'External Department';
    const aspect = collab.aspect_title || 'Specialist Domain';
    const aspectId = (collab.aspect_id || '').toLowerCase();
    const aff = collab.aspect_affinity_score || 0.0;
    const suit = collab.collaboration_suitability_score || 0.0;

    let action = `expands cross-disciplinary perspectives in ${aspect}`;
    let pedagogy = `delivering guest masterclasses and student project critique`;

    if (aspectId.includes('law') || aspectId.includes('policy') || aspectId.includes('ethics')) {
      action = `bridges critical regulatory frameworks, statutory compliance, and ethical accountability`;
      pedagogy = `delivering guest workshops on professional standards, legal governance, and risk mitigation`;
    } else if (aspectId.includes('robotics') || aspectId.includes('hardware')) {
      action = `connects software/theoretical concepts with embedded hardware, sensor integration, and real-world physical systems`;
      pedagogy = `running applied lab demonstrations and hands-on cyber-physical case studies`;
    } else if (aspectId.includes('sustainability') || aspectId.includes('climate')) {
      action = `embeds environmental sustainability, carbon lifecycle metrics, and circular design principles`;
      pedagogy = `leading specialized masterclasses on eco-standards and sustainable industry practices`;
    } else if (aspectId.includes('computational') || aspectId.includes('modeling') || aspectId.includes('simulation')) {
      action = `deepens rigorous computational modeling, numerical optimization, and formal algorithmic simulation`;
      pedagogy = `delivering advanced seminars on quantitative modeling and algorithmic benchmarking`;
    } else if (aspectId.includes('ux') || aspectId.includes('human') || aspectId.includes('design')) {
      action = `introduces human-centered design heuristics, usability testing, and accessibility frameworks`;
      pedagogy = `mentoring student teams on user research and interface prototyping`;
    } else if (aspectId.includes('business') || aspectId.includes('economics') || aspectId.includes('management')) {
      action = `integrates market feasibility, economic appraisal, and technology commercialization perspectives`;
      pedagogy = `co-facilitating business case analysis and industry translation sessions`;
    } else if (aspectId.includes('health') || aspectId.includes('biomedical')) {
      action = `provides clinical translation, physiological data integration, and biomedical device compliance`;
      pedagogy = `presenting clinical impact studies and healthcare deployment challenges`;
    }

    return `<strong>${pos}</strong> from <em>${dept}</em> who ${action}. Complements ${leadName} (${leadDept}) by ${pedagogy} (Suitability: <strong>${suit.toFixed(2)}</strong>, Aspect Affinity: <strong>${aff.toFixed(2)}</strong>).`;
  }
}

// Instantiate on DOM load
document.addEventListener('DOMContentLoaded', () => {
  window.recommenderApp = new RecommenderApp();
});
