"use strict";

const Workspace = (() => {
  const clone = value => JSON.parse(JSON.stringify(value));
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const genres = ['ballad', 'classical', 'rock', 'dance', 'kpop', 'jpop', 'newage', 'hiphop', 'piano_rock', 'pop_punk', 'hard_rock', 'emo_rock', 'modern_rock'];
  const voices = ['female', 'mezzo', 'female_low', 'male_low'];
  const choices = { melody_role: ['vocal', 'lead_guitar', 'acoustic_guitar', 'keys'], form: ['standard', 'chorus_first'],
    seed_role: ['chorus', 'verse', 'intro'], melody_source: ['ai', 'own', 'rest'], intro_player: ['guitar', 'vocal'] };
  const flags = ['fit_voice', 'key_change', 'melody_shape'];
  function validatePatch(patch) {
    if (!patch || typeof patch !== 'object' || Array.isArray(patch)) throw new Error('설정 변경 형식이 잘못됐습니다.');
    for (const [key, value] of Object.entries(patch)) {
      const valid = key === 'tempo' ? Number.isInteger(value) && value >= 40 && value <= 200
        : key === 'genre' ? genres.includes(value)
        : key === 'voice' ? voices.includes(value)
        : key === 'lyrics_mode' ? ['ah', 'model'].includes(value)
        : key === 'key_shift' ? Number.isInteger(value) && [0, -1, -2, -3, -4].includes(value)
        : key in choices ? choices[key].includes(value) : flags.includes(key) && typeof value === 'boolean';
      if (!valid) throw new Error('지원하지 않거나 범위를 벗어난 설정입니다: ' + key);
    }
    return clone(patch);
  }
  function createEditor(adapter) {
    const history = [];
    return {
      apply(patch, expected = null) {
        const checked = validatePatch(patch);
        const before = clone(adapter.read());
        if (expected && !same(before, expected)) throw new Error('그 사이 설정이 바뀌었습니다. 현재 설정으로 다시 요청해 주세요.');
        const after = { ...before, ...checked };
        adapter.write(after, 'apply', Object.keys(checked));
        history.push({ before, after: clone(adapter.read()) });
        if (history.length > 10) history.shift();
        return { before, after: clone(adapter.read()) };
      },
      undo() {
        const entry = history[history.length - 1];
        if (!entry) return false;
        if (!same(adapter.read(), entry.after)) throw new Error('이후 직접 바꾼 설정이 있어 자동 되돌리기를 중단했습니다.');
        adapter.write(clone(entry.before), 'undo', []);
        history.pop();
        return true;
      },
      get canUndo() { return history.length > 0; },
    };
  }
  function isSettingRequest(text, mode, hasFile) {
    if (hasFile) return false;
    return Boolean(text.trim());
  }
  function requiresConfirmation(result) {
    return result.source !== 'rules' || result.confirmation_required !== false;
  }
  function pageForStep(step) { return Number(step) >= 3 ? 'result' : Number(step) === 2 ? 'make' : 'input'; }
  function nextPage(page, hasMelody, hasResult) {
    if (page === 'input') return hasMelody ? 'make' : hasResult ? 'result' : 'input';
    if (page === 'settings') return 'make';
    return hasResult ? 'result' : page;
  }
  return { createEditor, validatePatch, genres, voices, flags, isSettingRequest, requiresConfirmation, pageForStep, nextPage };
})();
if (typeof module !== "undefined" && module.exports) module.exports = Workspace;

if (typeof document !== 'undefined') {
  (function bootWorkspace() {
    const byId = id => document.getElementById(id);
    const labels = { tempo: '템포', genre: '장르', voice: '보컬 음색', lyrics_mode: '가사 방식', key_shift: 'YuE2 키 낮춤',
      melody_role: '멜로디 담당', form: '곡 구성', seed_role: '입력 멜로디 위치', melody_source: '나머지 멜로디',
      intro_player: '인트로 담당', fit_voice: '보컬 음역 맞춤', key_change: '마지막 후렴 전조', melody_shape: '새 구간 다듬기' };
    const controlIds = { melody_role: 'melody-role', form: 'form-choice', seed_role: 'seed-role',
      melody_source: 'melody-source', intro_player: 'intro-player', fit_voice: 'fit-voice', key_change: 'key-change', melody_shape: 'melody-shape' };
    const choiceNames = { vocal: '보컬', lead_guitar: '일렉 기타', acoustic_guitar: '어쿠스틱 기타', keys: '피아노',
      standard: '기본 구성', chorus_first: '후렴 먼저', chorus: '후렴', verse: '벌스', intro: '인트로',
      ai: '모델 제안', own: '규칙 변형', rest: '비워 두기', guitar: '리드 기타' };
    const voiceLabels = { female: '맑은 여성', mezzo: '따뜻한 여성', female_low: '저음 여성', male_low: '중저음 남성' };
    let commandMode = 'create', pending = false;
    const messages = [];
    const read = () => ({
      tempo: state.tempo, genre: state.genre,
      voice: byId('yuesong-voice').value,
      lyrics_mode: byId('yuesong-random').checked ? 'model' : 'ah',
      key_shift: Number(byId('yuesong-key').value),
      src: { ...state.src }, guide_voice: byId('vocal-style').value, style: state.style,
      guide_voice_source: byId('vocal-style').dataset.userSet || '',
      ...Object.fromEntries(Object.entries(controlIds).map(([key, id]) => [key, Workspace.flags.includes(key) ? byId(id).checked : byId(id).value])),
      control_sources: Object.fromEntries(Object.entries(controlIds).map(([key, id]) => [key, byId(id).dataset.userSet || ''])),
    });
    function write(settings, mode, keys) {
      state.tempo = settings.tempo;
      state.genre = settings.genre;
      byId('yuesong-voice').value = settings.voice;
      byId('yuesong-random').checked = settings.lyrics_mode === 'model';
      byId('yuesong-key').value = String(settings.key_shift);
      for (const [key, id] of Object.entries(controlIds)) {
        if (Workspace.flags.includes(key)) byId(id).checked = settings[key]; else byId(id).value = settings[key];
        if (mode === 'undo') byId(id).dataset.userSet = settings.control_sources[key];
        else if (keys.includes(key)) byId(id).dataset.userSet = '1';
      }
      syncIntroPlayer(); syncRandomLyrics();
      if (mode === 'undo') {
        state.src = { ...settings.src };
        state.style = settings.style;
        byId('vocal-style').value = settings.guide_voice;
        byId('vocal-style').dataset.userSet = settings.guide_voice_source;
      } else {
        if (keys.includes('tempo')) state.src.tempo = 'manual';
        if (keys.includes('genre')) { state.src.genre = 'rec'; state.src.instr ||= 'rec'; }
        if (keys.includes('voice')) {
          byId('vocal-style').value = settings.voice === 'male_low' ? 'male' : 'female';
          byId('vocal-style').dataset.userSet = '1';
        }
        // Genre/tempo edits invalidate the old StyleAttrs, matching the existing manual controls.
        // Voice/lyrics/key changes preserve unrelated style selections.
        if (keys.includes('genre') || keys.includes('tempo')) state.style = null;
      }
      syncStyleButtons();
      renderVoiceButtons(); renderChips();
      renderRequestSummary();
      const note = byId('workspace-changes');
      note.textContent = mode === 'undo' ? '도우미가 바꾼 설정을 되돌렸습니다. 입력 음표와 기존 음원은 그대로입니다.'
        : '설정이 바뀌었습니다. 입력 음표와 기존 음원은 그대로이며, 새로 만들 때 변경 설정이 적용됩니다.';
      note.classList.remove('hidden');
    }
    const editor = Workspace.createEditor({ read, write });
    const context = () => {
      const value = read();
      return { tempo: value.tempo || DEFAULT_BPM[value.genre] || 90, genre: value.genre,
        voice: value.voice, lyrics_mode: value.lyrics_mode, key_shift: value.key_shift,
        ...Object.fromEntries(Object.keys(controlIds).map(key => [key, value[key]])) };
    };
    function renderRequestSummary() {
      const value = context();
      byId('workspace-request-summary').textContent = `${value.tempo} BPM · ${GENRES[value.genre]} · 멜로디 ${choiceNames[value.melody_role]} · ${voiceLabels[value.voice]} · ${choiceNames[value.form]} · 입력은 ${choiceNames[value.seed_role]} · 나머지는 ${choiceNames[value.melody_source]}`;
    }
    function say(text, who = 'app') {
      document.body.dataset.conversationStarted = 'true';
      chatSay(who, String(text));
      messages.push({ role: who === 'me' ? 'user' : 'assistant', content: String(text).slice(0, 500) });
      if (messages.length > 10) messages.shift();
    }
    function updateUndo() { byId('workspace-undo').disabled = !editor.canUndo; }
    function apply(patch, expected = null) {
      const change = editor.apply(patch, expected);
      updateUndo();
      return change;
    }
    function setCommandMode(mode) {
      commandMode = mode;
      byId('workspace-mode-create').setAttribute('aria-pressed', String(mode === 'create'));
      byId('workspace-mode-settings').setAttribute('aria-pressed', String(mode === 'settings'));
      byId('chat-text').placeholder = mode === 'settings' ? '예: BPM을 110으로 바꿔줘 / 따뜻한 음색 추천해줘' : '예: 템포 100으로 바꿔줘 (장르·보컬은 위 버튼으로)';
    }
    function openDock(open = true) {
      const dialog = byId('workspace-assistant-dialog');
      if (open && document.body.dataset.workspacePage !== 'input') {
        dialog.append(byId('assistant-dock'));
        if (!dialog.open) dialog.showModal();
      } else if (!open && dialog.open) dialog.close();
      byId('workspace-chat-toggle').setAttribute('aria-expanded', String(open));
      if (open) byId('chat-text').focus();
    }
    byId('workspace-dialogue').append(byId('assistant-dock'));
    byId('workspace-section-heading').append(byId('workspace-edit-settings'));
    // 대화 옵션 button removed (user, 2026-10-19: YuE2 makes the whole song); its dialog stays for 고급 도구's rebuild
    byId('workspace-options-close').onclick = () => byId('workspace-options-dialog').close();
    byId('workspace-dialogue-close').onclick = () => openDock(false);
    byId('workspace-assistant-dialog').addEventListener('close', () => {
      byId('workspace-dialogue').append(byId('assistant-dock'));
      byId('workspace-chat-toggle').setAttribute('aria-expanded', 'false');
    });
    const oldSetView = setView;
    const oldGoStep = goStep;
    const oldNextStep = nextStep;
    function setPage(page) {
      if (page === 'settings') page = 'input';
      const pageChanged = document.body.dataset.workspacePage !== page;
      document.body.dataset.workspacePage = page;
      document.body.dataset.hasSong = String(Boolean(state.song));
      if (document.body.dataset.workspaceView === 'studio') {
        byId('workspace-title').textContent = page === 'result' ? (document.body.dataset.resultPanel === 'yue' ? 'YuE2 음원·악보 완성' : '악보·파트 연습') : '작곡·곡 완성';
      }
      byId('workspace-edit-settings').classList.toggle('hidden', !['make', 'result'].includes(page));
      for (const item of byId('workspace-pages').children) {
        if (item.dataset.page === page) item.setAttribute('aria-current', 'step'); else item.removeAttribute('aria-current');
      }
      const summary = byId('workspace-request-summary');
      summary.classList.toggle('hidden', page !== 'make');
      renderRequestSummary();
      byId('btn-step-prev').disabled = page === 'input';
      byId('btn-step-next').textContent = page === 'input' ? '확인하고 만들기로 →' : '악보·연습으로 →';
      byId('btn-step-next').disabled = page === 'result' && document.body.dataset.resultPanel !== 'yue';
      byId('btn-step-next').classList.toggle('hidden', page === 'make' && !state.song && !state.result);
      if (pageChanged) byId('steps').scrollTop = 0;
      if (page !== 'input') openDock(false);
    }
    function setResultPanel(panel) {
      document.body.dataset.resultPanel = panel;
      if (document.body.dataset.workspacePage === 'result') byId('btn-step-next').disabled = panel !== 'yue';
      if (document.body.dataset.workspacePage === 'result') byId('workspace-title').textContent = panel === 'yue' ? 'YuE2 음원·악보 완성' : '악보·파트 연습';
      document.querySelectorAll('button[data-result-panel]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.resultPanel === panel)));
    }
    const titles = {
      studio: ['작업실', '만들기와 결과 확인을 한곳에서. 바꾸고 싶은 것은 도우미에게 말해 주세요.'],
      voices: ['목소리', '원하는 음색 방향을 고르고, 세부 조건은 도우미에게 맡기세요.'],
      advanced: ['고급 도구', '음표·리듬·편곡을 직접 고치고 싶을 때만 열어 보세요.'],
    };
    function chooseView(view) {
      document.body.dataset.workspaceView = view;
      oldSetView(['library', 'practice', 'queue', 'settings'].includes(view) ? view : 'create');
      for (const a of document.querySelectorAll('[data-workspace-view]')) {
        if (a === document.body) continue;
        a.classList.toggle('on', a.dataset.workspaceView === view);
        if (a.dataset.workspaceView === view) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
      }
      byId('workspace-create').classList.toggle('hidden', view !== 'create');
      byId('workspace-voice-panel').classList.toggle('hidden', view !== 'voices');
      const heading = byId('workspace-section-heading');
      heading.classList.toggle('hidden', !titles[view]);
      if (titles[view]) { byId('workspace-title').textContent = titles[view][0]; byId('workspace-description').textContent = titles[view][1]; }
      if (view === 'advanced') { oldGoStep(1); setStart('type'); byId('workspace-style-settings').open = true; setCommandMode('settings'); }
      if (view === 'create') {
        oldGoStep(1);
        byId('step-hint').textContent = '멜로디와 원하는 느낌을 알려 주세요. 준비되면 요청을 확인해요.';
      }
      if (view === 'studio' && currentStep() === 1) oldGoStep(2);
      setPage(['create', 'studio'].includes(view) ? Workspace.pageForStep(currentStep()) : 'auxiliary');
    }
    // The original controllers still own generation; navigation follows their completed stage.
    goStep = function (n, hint = '') {
      oldGoStep(n, hint);
      if (Number(n) >= 2) byId('workspace-compose-controls').open = Number(n) === 2;
      chooseView(Number(n) >= 2 ? 'studio' : 'create');
      if (Number(n) >= 3) setPage('result');
    };
    nextStep = function () {
      const page = document.body.dataset.workspacePage;
      if (page === 'result' && document.body.dataset.resultPanel === 'yue') {
        if (state.composing || !state.song) return showError('연습할 편곡 악보가 아직 없어요. 작곡이 끝난 뒤 이동해 주세요.');
        if (!yueBusy && yueSongResult && yueSongSource === state.song) adoptYueSong();
        hideError(); setResultPanel('practice');
        if (yueBusy) byId('practice-status').textContent = 'YuE2 전곡 생성은 계속 진행 중이에요. 지금은 현재 편곡 악보를 보고 있어요.';
        return;
      }
      if (page === 'input' && byId('steps').dataset.start === 'score' && !importedScore) {
        showError('악보 인식이 끝나야 다음으로 갈 수 있어요. 파일 옆 진행 상태를 확인해 주세요.');
        return;
      }
      if (page === 'input' && importedScore && !scoreAccepted('import', importedScore)) return;
      const next = Workspace.nextPage(page, Boolean(state.tokens.length || importedScore), Boolean(state.result || state.song));
      if (page === 'input' && next === 'make') {
        if (!state.src.genre || (melodyRole() === 'vocal' && !byId('vocal-style').value)) {
          hideError(); setStart('talk');
          say(!state.src.genre ? '어떤 장르로 만들까요? 예: 발라드, 여성 보컬. 대화에 답하고 제안된 변경안을 적용해 주세요.' : '보컬 음색을 대화로 알려 주세요. 예: 따뜻한 여성 목소리. 제안된 변경안을 적용하면 다음으로 갈 수 있어요.');
          byId('chat-text').focus(); return;
        }
        if (importedScore) setStart('score'); goStep(2);
        return;
      }
      if (page === 'make' && next === 'result') {
        if (state.composing) return showError('작곡이 끝난 뒤 악보·연습으로 이동해 주세요.');
        if (!yueBusy && yueSongResult && yueSongSource === state.song) adoptYueSong();
        oldGoStep(3); chooseView('studio'); setPage('result'); setResultPanel('practice'); return;
      }
      if (page === 'input') { say('먼저 악보를 첨부하거나 허밍으로 멜로디를 넣어 주세요.'); return; }
      oldNextStep();
    };
    byId('btn-step-next').onclick = () => nextStep();
    byId('workspace-edit-settings').onclick = () => { setCommandMode('settings'); openDock(); };
    byId('btn-step-prev').onclick = () => {
      const page = document.body.dataset.workspacePage;
      if (page === 'settings') { goStep(1); return; }
      if (page === 'make') { chooseView('create'); setStart('talk'); return; }
      goStep(2);
    };
    setView = function (view) {
      chooseView(view === 'create' && (state.result || state.song) ? 'studio' : view);
    };
    function renderSeedSummary() {
      const count = state.tokens.filter(t => t.degree > 0).length;
      byId('workspace-seed-summary').textContent = importedScore ? `올린 악보 ${importedScore.bars}마디 · ${importedScore.notes.length}음 · 입력을 보존하고 있어요.` : count ? `입력 멜로디 ${count}음 · 그대로 듣거나 필요한 음만 고칠 수 있어요.` : '멜로디를 아직 넣지 않았어요.';
      // the create screen's 입력 멜로디 듣기 / 음표 직접 고치기 were removed (user, 2026-10-19): notes are edited in 고급 도구
    }
    function renderVoiceButtons() {
      const container = byId('workspace-voice-options');
      container.replaceChildren();
      for (const [voice, label] of Object.entries(voiceLabels)) {
        const button = document.createElement('button');
        button.type = 'button'; button.textContent = label;
        button.setAttribute('aria-pressed', String(byId('yuesong-voice').value === voice));
        button.onclick = () => { try { apply({ voice }); say(label + ' 음색을 선택했습니다. 기존 음원은 바뀌지 않습니다.'); } catch (e) { say(e.message); } };
        container.append(button);
      }
    }
    function showProposal(patch, expected) {
      const li = document.createElement('li'); li.className = 'workspace-proposal';
      const list = document.createElement('dl');
      for (const [key, value] of Object.entries(patch)) {
        const dt = document.createElement('dt'), dd = document.createElement('dd');
        dt.textContent = labels[key];
        dd.textContent = key === 'voice' ? voiceLabels[value] : key === 'genre' ? GENRES[value] : key === 'lyrics_mode' ? (value === 'model' ? '모델 가사' : '아~') : typeof value === 'boolean' ? (value ? '켜기' : '끄기') : choiceNames[value] || String(value);
        list.append(dt, dd);
      }
      const button = document.createElement('button'); button.type = 'button'; button.textContent = '이 변경안 적용';
      button.onclick = () => {
        try { apply(patch, expected); button.disabled = true; button.textContent = '설정에 적용됨'; say('변경안을 설정에 적용했습니다. 음원 재생성은 시작하지 않았습니다.'); }
        catch (e) { say(e.message); }
      };
      li.append(list, button); byId('chat-log').append(li);
      byId('chat-log').scrollTop = byId('chat-log').scrollHeight;
    }
    async function request(text) {
      if (pending) return;
      pending = true; byId('chat-send').disabled = true;
      const expected = read();
      const history = messages.slice();
      say(text, 'me');
      try {
        const response = await fetch('/api/workspace/assistant', { method: 'POST', headers: { 'Content-Type': 'application/json', ...(window.instmateGeminiKey?.() ? { 'X-Gemini-Key': window.instmateGeminiKey() } : {}) },
          body: JSON.stringify({ text, context: context(), history }) });
        const result = await response.json().catch(() => ({ detail: '서버 응답이 비었어요(시간 초과이거나 연결이 끊겼어요). 잠시 뒤 다시 보내 주세요.' }));
        if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : `서버 응답 ${response.status}`);
        if (result.llm_note) say(result.llm_note);
        say((result.source === 'gemini' ? '[Gemini] ' : '') + (result.reply || '변경안을 확인해 주세요.'));
        const patch = Workspace.validatePatch(result.patch || {});
        if (Object.keys(patch).length) {
          if (Workspace.requiresConfirmation(result)) showProposal(patch, expected);
          else { apply(patch, expected); say('설정에 반영했습니다. 기존 음원은 그대로입니다.'); }
        }
      } catch (e) { say('설정을 변경하지 않았습니다: ' + e.message); }
      finally { pending = false; byId('chat-send').disabled = false; }
    }
    document.querySelectorAll('#views [data-workspace-view]').forEach(a => a.addEventListener('click', e => {
      e.preventDefault(); e.stopImmediatePropagation(); chooseView(a.dataset.workspaceView);
    }, true));
    document.querySelectorAll('.workspace-secondary [data-view]').forEach(a => a.addEventListener('click', e => {
      e.preventDefault(); e.stopImmediatePropagation(); chooseView(a.dataset.view);
    }, true));
    document.querySelectorAll('[data-workspace-open]').forEach(b => b.onclick = () => chooseView(b.dataset.workspaceOpen));
    document.querySelectorAll('[data-workspace-start]').forEach(b => b.onclick = () => {
      chooseView('create');
      if (b.dataset.workspaceStart === 'talk') { setStart('talk'); setCommandMode('create'); openDock(); return; }
      setStart(b.dataset.workspaceStart);
      if (b.dataset.workspaceStart === 'score') byId('import-file').click();
    });
    byId('workspace-mode-create').onclick = () => setCommandMode('create');
    byId('workspace-mode-settings').onclick = () => setCommandMode('settings');
    byId('workspace-chat-toggle').onclick = () => openDock(!byId('workspace-assistant-dialog').open);
    document.querySelectorAll('button[data-result-panel]').forEach(b => b.onclick = () => setResultPanel(b.dataset.resultPanel));
    // every genre and voice as a button that applies at once (no model needed; pressing it is the user's own confirmation)
    function renderChips() {
      const cur = read();
      const fill = (id, entries, picked, onPick) => {
        const box = byId(id); if (!box) return; box.replaceChildren();
        for (const [key, label] of entries) {
          const b = document.createElement('button'); b.type = 'button'; b.textContent = label;
          b.setAttribute('aria-pressed', String(picked === key));
          b.onclick = () => { try { onPick(key, label); } catch (e) { say(e.message); } };
          box.append(b);
        }
      };
      fill('chat-genre-chips', Workspace.genres.map(g => [g, GENRES[g] || g]), cur.genre,
        (g, label) => { apply({ genre: g }); say(label + ' 장르를 선택했습니다. 기존 음원은 바뀌지 않습니다.'); });
      fill('chat-voice-chips', Object.entries(voiceLabels), cur.voice,
        (v, label) => { apply({ voice: v }); say(label + ' 음색을 선택했습니다. 기존 음원은 바뀌지 않습니다.'); });
    }
    byId('chat-file').addEventListener('change', () => { byId('workspace-file-name').textContent = byId('chat-file').files[0]?.name || ''; });
    byId('workspace-rebuild').textContent = '만들기 화면 열기';
    byId('workspace-rebuild').onclick = () => { goStep(2); byId('workspace-options-dialog').close(); };
    byId('workspace-undo').onclick = () => {
      try { if (editor.undo()) say('도우미의 마지막 설정 변경을 되돌렸습니다.'); updateUndo(); }
      catch (e) { say(e.message); }
    };
    byId('chat-form').addEventListener('submit', async e => {
      if (pending) { e.preventDefault(); e.stopImmediatePropagation(); return; }
      const text = byId('chat-text').value.trim();
      const file = byId('chat-file').files[0];
      if (file) {
        e.preventDefault(); e.stopImmediatePropagation();
        if (chatBusy) return;
        chatBusy = true; byId('chat-send').disabled = true;
        try {
          say('악보를 읽고 있습니다: ' + file.name);
          await chatReadScore(file);
          renderSeedSummary();
          byId('chat-file').value = ''; byId('workspace-file-name').textContent = file.name;
          setStart('score');
          say('악보를 읽었습니다. 인식 결과를 확인하고 대화로 돌아가 원하는 설정을 알려 주세요.');
        } catch (err) { say('악보를 읽지 못했습니다: ' + err.message); }
        finally { chatBusy = false; byId('chat-send').disabled = false; }
        if (text) { byId('chat-text').value = ''; await request(text); }
        return;
      }
      if (!Workspace.isSettingRequest(text, commandMode, Boolean(byId('chat-file').files.length))) return;
      e.preventDefault(); e.stopImmediatePropagation();
      if (text) { byId('chat-text').value = ''; request(text); }
    }, true);
    new MutationObserver(renderSeedSummary).observe(byId('melody-view'), { childList: true });
    new MutationObserver(() => {
      if (!byId('error').classList.contains('hidden') && /장르를 아직 고르지|보컬을 아직 고르지/.test(byId('error').textContent)) {
        setPage('input'); setStart('talk'); say('장르와 보컬을 대화로 알려 주시고 변경안을 적용해 주세요.');
      }
    }).observe(byId('error'), { attributes: true, childList: true, subtree: true });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') openDock(false); });
    async function refreshStatus() {
      try {
        const response = await fetch('/api/workspace/assistant/status'); const status = await response.json();
        byId('workspace-connection').textContent = status.available ? `로컬 LLM 연결됨 · ${status.model} · 요청할 때 실행` : (status.reason || '언어 모델 연결 안 됨 · BPM 등 빠른 수정 가능');
      } catch (_) { byId('workspace-connection').textContent = '연결 확인 실패 · 서버 상태를 확인해 주세요.'; }
    }
    setMode('form'); chooseView('create'); setStart('talk'); setResultPanel('song'); setCommandMode('create');
    new MutationObserver(() => {
      if (document.body.dataset.workspaceView === 'studio') setPage(document.body.dataset.workspacePage);
    }).observe(byId('yuesong-status'), { childList: true });
    const banner = byId('public-demo-banner');
    if (banner) new ResizeObserver(() => document.body.style.setProperty('--public-notice-height', banner.getBoundingClientRect().height + 'px')).observe(banner);
    const playbackBars = [byId('libbar'), byId('playerbar')];
    const fitPlayback = () => {
      const playerHeight = playbackBars[1].getBoundingClientRect().height;
      playbackBars[0].style.bottom = playerHeight + 'px';
      document.body.style.setProperty('--playback-height', (playerHeight + playbackBars[0].getBoundingClientRect().height) + 'px');
    };
    const playbackObserver = new ResizeObserver(fitPlayback);
    playbackBars.forEach(bar => playbackObserver.observe(bar));
    fitPlayback();
    renderSeedSummary(); renderVoiceButtons(); renderChips(); refreshStatus();
    window.workspaceController = { editor, readSettings: read, context, setView: chooseView, request, setCommandMode, refreshStatus,
      openYue() { chooseView('studio'); oldGoStep(3); setPage('result'); setResultPanel('yue'); } };
  })();
}

// First run of the locally installed app: the model download card (GET /api/models/status, POST /api/models/yue2/download).
// It shows only when the server says it is a local app and the model is missing (a collapsed row, so the melody input stays in
// view); on the demo or any public server it stays hidden.  A lost connection never looks like a finished download, and a finished
// download refreshes the page's YuE2 availability.  The same status says which optional tools this install lacks (PDF, Demucs).
if (typeof document !== 'undefined' && document.getElementById('firstrun-card')) {
  (() => {
    const $ = id => document.getElementById(id);
    const card = $('firstrun-card'), summary = $('firstrun-summary'), terms = $('firstrun-terms'), start = $('firstrun-start'),
      bar = $('firstrun-bar'), pct = $('firstrun-pct'), msg = $('firstrun-msg');
    let timer = null, shown = false, wasRunning = false;
    const watch = ms => { if (!timer) timer = setInterval(poll, ms); };
    const unwatch = () => { if (timer) { clearInterval(timer); timer = null; } };
    const features = f => {
      if (!f) return;
      if ($('pdf-note')) $('pdf-note').classList.toggle('hidden', !!f.pdf);
      const note = $('practice-parts-note');
      if (note) note.classList.toggle('hidden', !!f.practice_parts);
      const sel = $('lib-part');
      if (sel) for (const o of sel.options) {
        if (o.value === 'vocal') continue;
        o.disabled = !f.practice_parts;
        o.textContent = o.textContent.replace(' (추가 도구 필요)', '') + (f.practice_parts ? '' : ' (추가 도구 필요)');
      }
    };
    const render = s => {
      if (!s || !s.enabled) { card.classList.add('hidden'); shown = false; unwatch(); return; }
      const running = s.state === 'running';
      if (s.installed && !running) {                      // a verified model: the card goes away and the page learns it can sing now
        card.classList.add('hidden'); shown = false; unwatch();
        if (wasRunning) { wasRunning = false; if (window.loadYueStatus) window.loadYueStatus(); }
        return;
      }
      card.classList.remove('hidden'); shown = true; wasRunning = wasRunning || running;
      $('firstrun-rev').textContent = s.revision || '';
      bar.classList.toggle('hidden', !running);
      bar.value = s.percent || 0;
      pct.textContent = running ? `${s.percent || 0}% (${(s.bytes_done / 1e9).toFixed(1)} / ${(s.expected_bytes / 1e9).toFixed(1)}GB)` : '';
      start.disabled = running || !terms.checked;
      terms.disabled = running;
      summary.textContent = running ? (s.phase === 'verify' ? '노래 모델 · 받은 파일을 확인하는 중' : `노래 모델 받는 중 ${s.percent || 0}%`)
        : s.error ? '노래 모델 · 받기 실패, 다시 눌러 주세요' : '노래 모델 받기 (약 4.5GB) · 악보·MIDI는 지금 바로 돼요';
      msg.textContent = s.error || (running ? (s.phase === 'verify' ? '받은 파일이 맞는지 확인하는 중이에요(1분쯤 걸려요).'
        : '받는 중이에요. 이 브라우저 창은 닫아도 되지만 앱(터미널 창)을 닫으면 멈춰요. 인터넷이 끊기면 멈추고, 다시 누르면 이어서 받아요.') : '');
      if (s.error || running) card.open = true;
      if (running) watch(2000); else unwatch();
    };
    async function poll() {
      try {
        const r = await fetch('/api/models/status');
        if (!r.ok) throw new Error('status ' + r.status);
        const s = await r.json();
        features(s.features);
        render(s);
      } catch (e) {
        if (!shown) { card.classList.add('hidden'); return; }       // the demo page and a missing server: stays hidden
        summary.textContent = '노래 모델 · 앱과 연결을 확인하는 중';   // it was showing: keep it, say what is happening, keep asking
        msg.textContent = '앱과 연결이 끊겼어요. 앱(터미널 창)이 켜져 있는지 확인해 주세요. 다시 연결되면 이어서 보여 드려요.';
        card.open = true; watch(3000);
      }
    }
    terms.onchange = () => { start.disabled = !terms.checked; };
    start.onclick = async () => {
      start.disabled = true;
      try {
        const r = await fetch('/api/models/yue2/download', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ accept_terms: terms.checked }) });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) { msg.textContent = typeof body.detail === 'string' ? body.detail : `서버 응답 ${r.status}`; start.disabled = !terms.checked; return; }
        render(body);
      } catch (e) { msg.textContent = '서버에 연결하지 못했어요. 앱이 켜져 있는지 확인해 주세요.'; start.disabled = !terms.checked; }
    };
    poll();
  })();
}
// 사용 안내 panel (user, 2026-10-19: usage help only, on the right): a line for the current step and fixed answers to
// common questions.  No language model, so it answers even while YuE2 runs.  Opened by the 씨앗이 button (user,
// 2026-10-19: "캐릭터 눌러서 하는 챗봇"); 화면 설정 → 사용 안내 hides the button.
if (typeof document !== 'undefined' && document.getElementById('guide-panel')) {
  (() => {
    const GUIDE_KEY = 'instmate-guide';
    const panel = document.getElementById('guide-panel'), now = document.getElementById('guide-now');
    const answer = document.getElementById('guide-answer'), btn = document.getElementById('btn-guide');
    const mascot = document.getElementById('mascot-btn');
    const NOW = {
      input: '멜로디를 넣어 주세요. 대화창의 악보 첨부나 허밍·녹음을 쓰고, 장르와 보컬(여성·남성)을 말해 주세요.',
      make: '요청을 확인하고 만들기를 눌러 주세요. 곡 초안은 보통 몇 초 안에 나와요.',
      result: '악보를 듣고 파트별로 연습해 보세요. YuE2 음원은 한 곡에 약 10분 걸려요.',
      advanced: '음표·리듬을 직접 고치는 곳이에요. 피아노롤에서 음을 찍거나 악보 파일을 불러오세요.',
      library: '만든 곡이 여기 모여요. 곡을 열면 다시 듣고 고칠 수 있어요.',
    };
    const FAQ = [
      ['어떤 장르가 있어?', '발라드, 록, 모던록, 감성록, 피아노 록, 팝펑크, 하드록, 제이팝, 케이팝이 있어요. 대화창에 "감성록으로"처럼 말하면 돼요.'],
      ['멜로디는 어떻게 넣어?', '대화창 아래 악보 첨부(MIDI·MusicXML·PDF)나 허밍·녹음을 쓰세요. 음을 하나씩 찍으려면 왼쪽 고급 도구로 가세요.'],
      ['음표는 어디서 고쳐?', '왼쪽 메뉴의 고급 도구에서 피아노롤로 직접 고칠 수 있어요.'],
      ['YuE2 음원은 얼마나 걸려?', '한 곡에 약 10분이에요. 한 번에 한 곡만 만들 수 있어서, 다른 사람이 만드는 중이면 끝난 뒤 다시 눌러 주세요.'],
      ['만든 곡은 어디 있어?', '왼쪽 메뉴의 내 곡에 모여요.'],
      ['YuE2 음원을 어디에 써도 돼?', 'YuE2 음원은 비상업(CC BY-NC 4.0) 조건이라 수업·연습·비영리 용도로만 써 주세요. 받은 파일 이름과 정보에 이 조건이 적혀 있어요.'],
      ['상업적으로 쓰고 싶어', 'YuE2 음원은 가이드로만 들어 주세요. 고급 도구에서 악기별 악보를 직접 편곡하고, 배포 모드(상업용으로 쓸 수 있는 악기 소리만)로 만들거나 직접 연주·녹음하세요. 상업 이용이 괜찮은지는 최종적으로 직접 확인해 주세요.'],
    ];
    const faq = document.getElementById('guide-faq');
    for (const [q, a] of FAQ) {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = q;
      b.onclick = () => { answer.textContent = a; };
      faq.append(b);
    }
    const update = () => {
      const view = document.body.dataset.workspaceView, page = document.body.dataset.workspacePage;
      now.textContent = NOW[view] || NOW[page] || NOW.input;
    };
    const open = on => {
      panel.hidden = !on;
      mascot.setAttribute('aria-expanded', String(on));
      if (on) { update(); panel.querySelector('button')?.focus(); }
    };
    const allow = on => {                                    // the 화면 설정 switch: the button shown or not
      mascot.hidden = !on;
      btn?.setAttribute('aria-pressed', String(on));
      if (!on) open(false);
      try { localStorage.setItem(GUIDE_KEY, on ? 'on' : 'off'); } catch (e) { /* not kept: it still works on this page */ }
    };
    let kept = null;
    try { kept = localStorage.getItem(GUIDE_KEY); } catch (e) { kept = null; }
    allow(kept !== 'off');
    open(false);
    if (btn) btn.onclick = () => allow(mascot.hidden);
    // Drag 씨앗이 anywhere; the spot (px from the right/bottom edges) is kept per viewer.
    const POS_KEY = 'instmate-mascot-pos', root = document.documentElement;
    const place = (x, y) => {
      const s = mascot.offsetWidth || 96;
      x = Math.max(0, Math.min(innerWidth - s, x)); y = Math.max(0, Math.min(innerHeight - s, y));
      root.style.setProperty('--mx', x + 'px'); root.style.setProperty('--my', y + 'px');
      return [x, y];
    };
    try { const p = JSON.parse(localStorage.getItem(POS_KEY)); if (p) place(p[0], p[1]); } catch (e) {}
    let drag = null, moved = false;
    mascot.addEventListener('dragstart', e => e.preventDefault());
    mascot.addEventListener('pointerdown', e => {
      const r = mascot.getBoundingClientRect();
      drag = { dx: e.clientX - r.right, dy: e.clientY - r.bottom, x0: e.clientX, y0: e.clientY }; moved = false;
      mascot.setPointerCapture(e.pointerId);
    });
    mascot.addEventListener('pointermove', e => {
      if (!drag) return;
      if (Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) > 6) moved = true;
      if (moved) drag.pos = place(innerWidth - e.clientX + drag.dx, innerHeight - e.clientY + drag.dy);
    });
    mascot.addEventListener('pointerup', () => {
      if (moved && drag.pos) try { localStorage.setItem(POS_KEY, JSON.stringify(drag.pos)); } catch (e) {}
      drag = null;
    });
    addEventListener('resize', () => { const cs = getComputedStyle(root); if (cs.getPropertyValue('--mx')) place(parseFloat(cs.getPropertyValue('--mx')), parseFloat(cs.getPropertyValue('--my'))); });
    mascot.onclick = () => { if (moved) { moved = false; return; } open(panel.hidden); };
    document.getElementById('guide-close').onclick = () => { open(false); mascot.focus(); };
    panel.addEventListener('keydown', e => { if (e.key === 'Escape') { open(false); mascot.focus(); } });
    // poses from the user's sheet (GCIM_1): headphones while YuE2 makes a song, a jump when it is done, asleep after
    // two idle minutes; otherwise the strumming frames.  The cancel button is shown exactly while a YuE2 take runs.
    const poseImg = mascot.querySelector('.m-pose'), cancel = document.getElementById('btn-yuesong-cancel');
    const TITLE = '씨앗이에게 사용법 묻기';
    let pose = '', poseTimer = 0, idleTimer = 0;
    const setPose = (name, label) => {
      pose = name;
      mascot.dataset.pose = name;
      poseImg.hidden = !name;
      if (name) poseImg.src = `/static/mascot/${name}.png`;
      mascot.title = label || TITLE;
    };
    const running = () => !!cancel && !cancel.classList.contains('hidden');
    const rest = () => setPose(running() ? 'headphones' : '', running() ? 'YuE2가 곡을 만드는 중이에요' : '');
    const wake = () => {
      clearTimeout(idleTimer);
      if (pose === 'sleep') rest();
      idleTimer = setTimeout(() => { if (!pose) setPose('sleep', '쉬는 중 · 누르면 깨어나요'); }, 120000);
    };
    if (cancel) new MutationObserver(() => {
      const was = pose === 'headphones';
      if (running()) { clearTimeout(poseTimer); setPose('headphones', 'YuE2가 곡을 만드는 중이에요'); }
      else if (was) {
        const failed = !document.getElementById('error')?.classList.contains('hidden');
        setPose(failed ? '' : 'jump', failed ? '' : '곡이 나왔어요!');
        poseTimer = setTimeout(rest, 4000);
      }
    }).observe(cancel, { attributes: true, attributeFilter: ['class'] });
    for (const ev of ['pointerdown', 'keydown']) document.addEventListener(ev, wake, { passive: true });
    // 씨앗이 speaks (user, 2026-10-19: "가이드는 씨앗이가 다 하는 거니까"): the greeting after the first-visit notice,
    // and every step hint, in a bubble kept up long enough to read (at least 5 s, ~110 ms a character).
    const bubble = document.getElementById('mascot-say'), hint = document.getElementById('step-hint');
    const queue = [];
    let talking = 0, last = '';
    const next = () => {
      clearTimeout(talking);
      const text = queue.shift();
      if (!text) { bubble.hidden = true; talking = 0; return; }
      bubble.textContent = text; bubble.hidden = false;
      talking = setTimeout(next, Math.max(5000, text.length * 110));
    };
    const say = (...lines) => { queue.push(...lines.filter(Boolean)); if (!talking) next(); };
    bubble.onclick = next;                                   // a click moves on
    const greet = () => say('안녕하세요! 저는 씨앗이예요.', '멜로디가 담긴 악보를 붙이거나 허밍해 주세요. 장르와 보컬(여성·남성)도 말해 주세요.',
      '도움이 필요하면 저를 눌러 주세요.');
    const terms = document.getElementById('terms-dialog');
    setTimeout(() => {                                       // after the notice script below has had its turn to open
      if (terms?.open) terms.addEventListener('close', greet, { once: true }); else greet();
    }, 0);
    if (hint) new MutationObserver(() => {
      const text = hint.textContent.trim();
      if (text && text !== last && !text.startsWith('멜로디와 원하는 느낌을')) say(text);   // the default line repeats the greeting
      last = text;
    }).observe(hint, { childList: true, characterData: true, subtree: true });
    wake();
    new MutationObserver(update).observe(document.body, { attributes: true, attributeFilter: ['data-workspace-view', 'data-workspace-page'] });
  })();
}

// 시작하기 전에 (user, 2026-10-19): the YuE2 non-commercial terms, the commercial route and upload rights, once per browser.
if (typeof document !== 'undefined' && document.getElementById('terms-dialog')?.showModal) {
  (() => {
    const KEY = 'instmate-terms-v1', dlg = document.getElementById('terms-dialog');
    let seen = null;
    try { seen = localStorage.getItem(KEY); } catch (e) { seen = null; }
    if (seen === 'ok') return;
    dlg.addEventListener('cancel', e => e.preventDefault());   // Esc does not skip it
    dlg.addEventListener('close', () => { try { localStorage.setItem(KEY, 'ok'); } catch (e) { /* asked again next visit */ } });
    dlg.showModal();
  })();
}

// "내 Gemini 키 쓰기": the key lives in this browser only (localStorage); an empty or odd value is refused here, the
// server checks it again.
if (typeof document !== 'undefined' && document.getElementById('gemini-box')) {
  (() => {
    const input = document.getElementById('gemini-key'), state = document.getElementById('gemini-state');
    const title = document.querySelector('#gemini-box > summary');
    const show = () => {
      const on = !!window.instmateGeminiKey?.();
      state.textContent = on ? '키를 쓰는 중이에요. 한도가 차면 qwen으로 넘어가요.' : '키가 없어요. 기본 규칙(과 qwen)으로 대화해요.';
      title.textContent = on ? 'Gemini 키 사용중' : '내 Gemini 키 쓰기 (선택)';
      document.getElementById('gemini-box').classList.toggle('gemini-on', on);
    };
    document.getElementById('gemini-save').onclick = () => {
      const key = input.value.trim();
      if (!/^[A-Za-z0-9._-]{20,128}$/.test(key)) { state.textContent = '키 모양이 맞지 않아요. 영문·숫자와 . _ - 만 쓰는 20자 이상이에요.'; return; }
      try { localStorage.setItem('instmate-gemini-key', key); } catch (e) { state.textContent = '이 브라우저는 저장을 막고 있어요.'; return; }
      input.value = ''; show();
    };
    document.getElementById('gemini-clear').onclick = () => { try { localStorage.removeItem('instmate-gemini-key'); } catch (e) { /* nothing stored */ } input.value = ''; show(); };
    show();
  })();
}
