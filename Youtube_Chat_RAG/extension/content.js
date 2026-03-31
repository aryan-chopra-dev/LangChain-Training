function getYouTubeVideoId() {
  const url = new URL(window.location.href);
  return url.searchParams.get("v");
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "GET_VIDEO_ID") {
    sendResponse({ videoId: getYouTubeVideoId() });
  }
});