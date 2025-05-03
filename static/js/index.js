function sendQuery() {
    let query = $('#query-input').val().trim();
    if (!query) {
        alert("Please enter a question.");
        return;
    }

    let smeMode = $('input[name="sme-selection"]:checked').val();
    let selectedTopics = [];

    if (smeMode === 'human') {
        selectedTopics = $('input[name="topics"]:checked').map((_, el) => el.value).get();
        if (selectedTopics.length === 0) {
            alert("Please select at least one topic for Human-Selected SME.");
            return;
        }
    }

    let useGeneralKnowledge = $('#include-general-knowledge').is(':checked');

    $('#chat-box').append(`<div class='message user'>User: ${escapeHtml(query)}</div>`);
    $('#query-input').val('');

    $.ajax({
        url: '/query',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            query: query,
            topics: selectedTopics,
            use_general_knowledge: useGeneralKnowledge
        }),
        success: function(response) {
            let formattedResponse = marked.parse(response.response);
            $('#chat-box').append(`<div class='message bot'><div class='markdown-body'>${formattedResponse}</div></div>`);
            MathJax.typesetPromise().then(scrollToBottom);
        },
        error: function(xhr) {
            alert("Error: " + (xhr.responseJSON?.error || "An error occurred."));
        }
    });
}
$('#query-input').keypress(function(e) {
    if (e.which === 13 && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
});

function clearChat() {
    $.post('/clear_chat', function(response) {
        $('#chat-box').empty();
        alert(response.message);
    });
}

function scrollToBottom() {
    $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function loadChatHistory() {
    $.get('/load_conversation', function(response) {
        const history = response.chat_history || [];
        $('#chat-box').empty();
        history.forEach(msg => {
            const roleClass = msg.role === 'user' ? 'user' : 'bot';
            const content = msg.role === 'user' ? escapeHtml(msg.content) : marked.parse(msg.content);
            const html = msg.role === 'user'
                ? `<div class='message ${roleClass}'>User: ${content}</div>`
                : `<div class='message ${roleClass}'><div class='markdown-body'>${content}</div></div>`;
            $('#chat-box').append(html);
        });
        MathJax.typesetPromise().then(scrollToBottom);
    });
}

function loadTopicsList() {
    $.get('/list_indexes', function(data) {
        const topicsDiv = $('#topics-list').empty();
        data.filter(index => index !== 'table_of_contents').forEach(topic => {
            topicsDiv.append(`<label><input type="checkbox" name="topics" value="${topic}"> ${topic}</label><br>`);
        });
    });
}

$(document).ready(function() {
    loadChatHistory();
    $('input[name="sme-selection"]').change(function() {
        if ($(this).val() === 'human') {
            $('#topic-selection').show();
            $('#general-knowledge-toggle').hide();
            loadTopicsList();
        } else {
            $('#topic-selection').hide();
            $('#general-knowledge-toggle').show();
        }
    });
});
