const API_URL = "http://127.0.0.1:5000";

const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const fileInput = document.getElementById("pdfInput");
const fileNameSpan = document.getElementById("fileName");

fileInput.addEventListener("change", function () {
  if (this.files && this.files.length > 0) {
    fileNameSpan.innerText = this.files[0].name;
  }
});

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  appendMessage(text, "user");
  userInput.value = "";

  const loadingId = appendMessage("Thinking...", "bot", true);

  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    const data = await response.json();

    removeMessage(loadingId);

    if (data.response) {
      appendMessage(data.response, "bot");
    }
  } catch (error) {
    console.error("Error:", error);
    removeMessage(loadingId);
    appendMessage(
      "Connection error. Please check if backend is running.",
      "bot",
    );
  }
}

async function uploadPDF() {
  if (fileInput.files.length === 0) {
    alert("Будь ласка, оберіть файл!");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const statusText = document.getElementById("uploadStatus");
  statusText.innerText = "Processing file... This may take a minute.";
  statusText.style.color = "blue";

  try {
    const response = await fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (response.ok) {
      statusText.innerText = "Success: " + result.message;
      statusText.style.color = "green";
    } else {
      statusText.innerText = "Error: " + result.error;
      statusText.style.color = "red";
    }
  } catch (error) {
    console.error("Upload error:", error);
    statusText.innerText = "Connection error.";
    statusText.style.color = "red";
  }
}

function appendMessage(text, sender, isLoading = false) {
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", sender);

  const msgId = "msg-" + Date.now();
  msgDiv.id = msgId;

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.innerHTML = text;

  if (isLoading) {
    bubble.style.fontStyle = "italic";
    bubble.style.opacity = "0.7";
  }

  msgDiv.appendChild(bubble);
  chatBox.appendChild(msgDiv);

  chatBox.scrollTop = chatBox.scrollHeight;

  return msgId;
}

function removeMessage(id) {
  const msg = document.getElementById(id);
  if (msg) {
    msg.remove();
  }
}

userInput.addEventListener("keypress", function (event) {
  if (event.key === "Enter") {
    sendMessage();
  }
});
