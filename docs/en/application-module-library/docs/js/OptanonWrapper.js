function OptanonWrapper() {
    if (typeof window._hsq === 'undefined') {
        window._hsq = [];
    }

    if (OnetrustActiveGroups.includes("C0004")) {
        window._hsq.push(['doNotTrack', {track: true}]); // enable tracking for user
    } else {
        window._hsq.push(['doNotTrack']); // turn off tracking for the user
    }
}