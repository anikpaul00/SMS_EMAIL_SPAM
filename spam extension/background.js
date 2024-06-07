chrome.action.onClicked.addListener(() => {
  chrome.windows.create({
    url: "https://smsemailspam-zkhu3f6vnacb8zeyv9mfan.streamlit.app/",
    type: "popup",
    width: 800,
    height: 600
  });
});