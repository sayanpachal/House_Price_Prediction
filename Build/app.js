function getBathValue() {
  var uiBathrooms = document.getElementsByName("uiBathrooms");
  for (let i = 0; i < uiBathrooms.length; i++) {
    if (uiBathrooms[i].checked) {
      return parseInt(uiBathrooms[i].value);
    }
  }
  return -1;
}

function getBHKValue() {
  var uiBHK = document.getElementsByName("uiBHK");
  for (let i = 0; i < uiBHK.length; i++) {
    if (uiBHK[i].checked) {
      return parseInt(uiBHK[i].value);
    }
  }
  return -1;
}

function onClickedEstimatePrice() {
  console.log("Estimate price button clicked");
  var sqft = document.getElementById("uiSqft");
  var bhk = getBHKValue();
  var bathrooms = getBathValue();
  var location = document.getElementById("uiLocations");
  var estPrice = document.getElementById("uiEstimatedPrice");


  var url = "http://127.0.0.1:5000/predict_home_price";  // For POST
  // var url = "/api/predict_home_price"; // Only use the applicable one

  $.post(url, {
      total_sqft: parseFloat(sqft.value),
      bhk: bhk,
      bath: bathrooms,
      location: location.value
  }, function(data, status) {
      if (data && data.estimated_price) {
        estPrice.innerHTML = `<h2>${data.estimated_price} Lakh</h2>`;
      } else {
        estPrice.innerHTML = "<h2>Error fetching price</h2>";
      }
      console.log(status);
  });
}

function onPageLoad() {
  console.log("document loaded");
  // var url = "/api/get_location_names";
var url = "http://127.0.0.1:5000/get_location_names"; 
  $.get(url, function(data, status) {
      console.log("got response for get_location_names request");
      if(data && data.locations) {
          var uiLocations = $('#uiLocations');
          uiLocations.empty();
          data.locations.forEach(function(loc) {
              var opt = new Option(loc);
              uiLocations.append(opt);
          });
      }
  });
}

window.onload = onPageLoad;
