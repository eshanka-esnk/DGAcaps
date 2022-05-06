$(function(){
    $('button#addButton').bind('click',function(){
      var $row = $(this).closest('tr');
      $row.addClass('selected').siblings().removeClass('selected');
      var value = $('table#datatable tr.selected td:first').html();
      $.getJSON('/getURL',{
          url: value
      });
      $('.jsflash').html("<div class='success'><span id='closebttn' class='closebtn'>&times;</span>success: Blacklisted domain.</div>");
      return false;
    });
  });

$(function(){
    $('.jsflash').bind('click',function(){
        $('.jsflash').empty()
        return false
    });
})