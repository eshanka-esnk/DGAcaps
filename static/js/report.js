var divclass
var message

$(function(){
    $('button#addButton').bind('click',function(){
      var $row = $(this).closest('tr');
      $row.addClass('selected').siblings().removeClass('selected');
      var value = $('table#datatable tr.selected td:first').html();
      $.getJSON('/getURL',{
          url: value
      }, function(data){
        if(data.result == 'error'){
          divclass = 'error'
          message = 'Domain already blacklisted'
          $('.jsflash').html("<div class='"+ divclass +"'><span id='closebttn' class='closebtn'>&times;</span>"+ divclass +":"+ message +"</div>");
          return false;
        }
        else if(data.result == 'success'){
          divclass = 'success'
          message = 'Blacklisted domain.'
          $('.jsflash').html("<div class='"+ divclass +"'><span id='closebttn' class='closebtn'>&times;</span>"+ divclass +":"+ message +"</div>");
          return false;
        }
        else{
          return false;
        }
      });
      return false;
    });
  });

$(function(){
    $('.jsflash').bind('click',function(){
        $('.jsflash').empty()
        return false
    });
})