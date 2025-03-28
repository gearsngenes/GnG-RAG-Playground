function validateTopicName() {
    let input = $('#new-index-name').val();
    $('#name-error').toggle(!/^[a-z0-9-]+$/.test(input));
}

function loadIndexes() {
    $.get('/list_indexes', function(data) {
        let indexSelect = $('#existing-index-select');
        indexSelect.empty().append('<option value="" disabled selected>Select a topic...</option>');

        if (data.length === 0) {
            indexSelect.append('<option value="" disabled>No topics available</option>');
            return;
        }

        data.forEach(index => {
            if (index !== "table_of_contents") {
                indexSelect.append(`<option value="${index}">${index}</option>`);
            }
        });
    }).fail(() => alert("Failed to load topics."));
}

function fetchDescription() {
    let indexName = $('#existing-index-select').val();
    if (!indexName) return;

    $.ajax({
        url: '/get_index_description',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ index_name: indexName }),
        success: function(data) {
            $('#edit-description').val(data.description).prop('disabled', false);
            $('#update-button').prop('disabled', false);
        },
        error: function(xhr) {
            alert("Failed to load description. Error: " + xhr.responseText);
        }
    });
}

function listUploadedFiles() {
    let indexName = $('#existing-index-select').val();
    if (!indexName) return;

    $.ajax({
        url: '/list_uploaded_files',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ index_name: indexName }),
        success: function(data) {
            let fileList = $('#file-list').empty();
            if (data.files.length > 0) {
                data.files.forEach(file => {
                    let statusText = file.embedded ? "Embedded" : "Not Embedded";
                    fileList.append(`
                        <li>
                            <input type="checkbox" value="${file.name}">
                            ${file.name} <span style="color: ${file.embedded ? 'green' : 'red'};">(${statusText})</span>
                        </li>
                    `);
                });
            } else {
                fileList.append("<li>No files uploaded.</li>");
            }
        },
        error: function(xhr) {
            alert("Error fetching files: " + xhr.responseText);
        }
    });
}

function unembedSelectedFiles() {
    let indexName = $('#existing-index-select').val();
    if (!indexName) {
        alert("Please select a topic first.");
        return;
    }

    let selectedFiles = $('#file-list input[type="checkbox"]:checked')
        .map((_, checkbox) => checkbox.value)
        .get();

    if (selectedFiles.length === 0) {
        alert("Please select at least one file to unembed.");
        return;
    }

    $.ajax({
        url: '/unembed_files',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ index_name: indexName, files: selectedFiles }),
        success: function(response) {
            alert(response.message);
            listUploadedFiles();  // Refresh embedding status
        },
        error: function(xhr) {
            alert("Error unembedding files: " + xhr.responseText);
        }
    });
}

function deleteSelectedFiles() {
    let indexName = $('#existing-index-select').val();
    if (!indexName) return;

    let selectedFiles = $('#file-list input[type="checkbox"]:checked')
        .map((_, checkbox) => checkbox.value)
        .get();

    if (selectedFiles.length === 0) {
        alert("Please select at least one file to delete.");
        return;
    }

    $.ajax({
        url: '/delete_files',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ index_name: indexName, files: selectedFiles }),
        success: function(response) {
            alert(response.message);
            listUploadedFiles(); // Refresh list
        },
        error: function(xhr) {
            alert("Error deleting files: " + xhr.responseText);
        }
    });
}

function embedSelectedFiles() {
    let indexName = $('#existing-index-select').val();
    let chunkSize = $('#chunk-size').val();
    if (!indexName) {
        alert("Please select a topic first.");
        return;
    }

    let selectedFiles = $('#file-list input[type="checkbox"]:checked')
        .map((_, checkbox) => checkbox.value)
        .get();

    if (selectedFiles.length === 0) {
        alert("Please select at least one file to embed.");
        return;
    }

    $.ajax({
        url: '/embed_files',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ index_name: indexName, files: selectedFiles, chunk_size: chunkSize }),
        success: function(response) {
            alert(response.message);
            listUploadedFiles(indexName);
        },
        error: function(xhr) {
            alert("Error embedding files: " + xhr.responseText);
        }
    });
}

function createTopic() {
    let indexName = $('#new-index-name').val();
    let description = $('#new-index-description').val();

    if (!indexName || !/^[a-z0-9-]+$/.test(indexName)) {
        alert("Invalid topic name! Only lowercase letters, numbers, and '-' are allowed.");
        return;
    }

    $.ajax({
        url: '/create_index',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ index_name: indexName, description: description }),
        success: function(response) {
            alert(response.message);
            $('#new-index-name').val('');
            $('#new-index-description').val('');
            loadIndexes();
        },
        error: function(xhr) {
            alert("Error: " + xhr.responseJSON.error);
        }
    });
}

function updateDescription() {
    let indexName = $('#existing-index-select').val();
    let newDescription = $('#edit-description').val();

    if (!indexName) return;

    $.ajax({
        url: '/update_index_description',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ index_name: indexName, description: newDescription }),
        success: function(response) {
            alert(response.message);
        },
        error: function(xhr) {
            alert("Error: " + xhr.responseJSON.error);
        }
    });
}

function deleteTopic() {
    let indexName = $('#existing-index-select').val();
    if (!indexName) {
        alert("Please select a topic to delete.");
        return;
    }

    $.ajax({
        url: '/delete_index',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ index_name: indexName }),
        success: function(response) {
            alert(response.message);
            loadIndexes();
            $('#file-list').empty();  // Clear file list after deletion
        },
        error: function(xhr) {
            alert("Error: " + xhr.responseJSON.error);
        }
    });
}

$('#file-input').on('change', function() {
    let file = this.files[0];
    if (!file) return;
    let ext = file.name.split('.').pop().toLowerCase();
    if (['jpg', 'jpeg', 'png'].includes(ext)) {
        $('#image-description-container').show();
    } else {
        $('#image-description-container').hide();
        $('#image-description').val(''); // Clear previous entry
    }
});

function uploadDocument() {
    let file = $('#file-input')[0].files[0];
    let indexName = $('#existing-index-select').val();
    if (!file || !indexName) {
        alert("Please select an index and a file.");
        return;
    }

    let ext = file.name.split('.').pop().toLowerCase();
    let description = $('#image-description').val().trim();
    if (['jpg', 'jpeg', 'png'].includes(ext) && !description) {
        alert("Please provide a description for the image.");
        return;
    }

    let formData = new FormData();
    formData.append('file', file);
    formData.append('index_name', indexName);
    if (description) {
        formData.append('image_description', description);
    }

    $.ajax({
        url: '/upload_document',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            alert(response.message);
            listUploadedFiles();
            $('#image-description-container').hide();
            $('#image-description').val('');
        },
        error: function(xhr) {
            alert("Error uploading file: " + xhr.responseText);
        }
    });
}

$(document).ready(loadIndexes);
