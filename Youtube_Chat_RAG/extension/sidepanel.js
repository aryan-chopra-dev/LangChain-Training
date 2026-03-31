async function getCurrentTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0];
}

async function getVideoIdFromTab(tabId) {
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(tabId, { type: "GET_VIDEO_ID" }, (response) => {
      resolve(response?.videoId || null);
    });
  });
}

function setStatus(text) {
  document.getElementById("status").textContent = text;
}

function setAnswer(text) {
  document.getElementById("answer").textContent = text;
}

document.getElementById("askBtn").addEventListener("click", async () => {
  setStatus("Checking current tab...");
  setAnswer("");

  const question = document.getElementById("question").value.trim();
  const videoLanguage = document.getElementById("lang").value.trim() || "en";

  if (!question) {
    setStatus("Please enter a question.");
    return;
  }

  const tab = await getCurrentTab();
  if (!tab || !tab.id) {
    setStatus("Could not detect active tab.");
    return;
  }

  const videoId = await getVideoIdFromTab(tab.id);

  if (!videoId) {
    setStatus("Open a YouTube video page first.");
    return;
  }

  setStatus("Asking backend...");

  chrome.runtime.sendMessage(
    {
      type: "ASK_BACKEND",
      payload: {
        video_id: videoId,
        video_language: videoLanguage,
        question: question
      }
    },
    (response) => {
      if (!response) {
        setStatus("No response from extension backend.");
        return;
      }

      if (!response.ok) {
        setStatus("Error");
        setAnswer(response.error || "Unknown error");
        return;
      }

      setStatus("Done");
      setAnswer(response.data.answer);
    }
  );
});